"""QQ群内 @机器人 -> DeepSeek 回复（QQ 官方 WebSocket + botpy）。"""

import asyncio
import logging
import os
import sqlite3
from collections import deque
from pathlib import Path
from weakref import WeakValueDictionary

import botpy
from botpy.message import GroupMessage
from dotenv import load_dotenv
from openai import AsyncOpenAI

from memory import ConversationMemory
from generation import GENERATION_TOKEN_BUDGET, configure_diagnostics, generate_completion
from web_search import (LINK_REQUEST_POLICY, SearchContext, SEARCH_ANSWER_POLICY, prepare_web_search,
                        render_search_answer, requested_links_fallback, wants_links)
from vision import DEFAULT_IMAGE_QUESTION, VISION_INSTRUCTIONS, VisionError, VisionInput, prepare_vision

load_dotenv()

QQ_APP_ID = os.getenv("QQ_APP_ID", "").strip()
QQ_APP_SECRET = os.getenv("QQ_APP_SECRET", "").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").strip().lower() not in {"0", "false", "off", "no"}
VISION_ENABLED = os.getenv("VISION_ENABLED", "true").strip().lower() not in {"0", "false", "off", "no"}
APPLICATION_VERSION = "3.2"
GROUP_MEMORY_INSTRUCTIONS = (
    "\n\n这是同一个QQ群内多位成员共享的对话历史。"
    "用户消息开头的【群成员标识】由程序提供，用于区分发言者。"
    "请回答当前发言者的问题，不要把其他成员的姓名、偏好或经历当成当前成员的。"
    "群成员标识仅用于内部区分，不要在回复中输出这些标识。"
)

missing = [
    name for name, value in (
        ("QQ_APP_ID", QQ_APP_ID),
        ("QQ_APP_SECRET", QQ_APP_SECRET),
        ("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY),
    ) if not value
]
if missing:
    raise RuntimeError("请先在 .env 文件中填写：" + ", ".join(missing))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deepseek_qq_at")

ai = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    timeout=120.0,
    max_retries=0,
)

# SDK 在网络重连后可能重复投递同一条事件，避免重复扣费/重复回复。
recent_message_ids = deque(maxlen=500)

PROJECT_DIR = Path(__file__).resolve().parent
# 本地人设不进入版本库；首次使用时读取公开示例。
PERSONA_PATH = PROJECT_DIR / "persona.txt"
if not PERSONA_PATH.is_file():
    PERSONA_PATH = PROJECT_DIR / "persona.example.txt"
SYSTEM_PERSONA = PERSONA_PATH.read_text(encoding="utf-8-sig").strip()
if not SYSTEM_PERSONA:
    raise RuntimeError("人设文件不能为空，请填写 persona.txt 或恢复 persona.example.txt")
memory_path = Path(os.getenv("MEMORY_DB_PATH", "data/memory.sqlite3"))
memory = ConversationMemory(PROJECT_DIR / memory_path, max_turns=30)
# 同一会话串行处理；没有等待者时锁可回收，避免用户数增加导致常驻内存增长。
conversation_locks = WeakValueDictionary()


class DeepSeekQQBot(botpy.Client):
    async def on_ready(self):
        log.info("QQ 机器人已连接：%s，应用版本 %s", getattr(self.robot, "name", "未知名称"), APPLICATION_VERSION)

    async def on_group_at_message_create(self, message: GroupMessage):
        if message.id in recent_message_ids:
            return
        recent_message_ids.append(message.id)

        # QQ @事件通常已移除开头的 @机器人 标签。
        question = (message.content or "").strip()
        log.info("收到群聊 @事件：group=%s message_id=%s", message.group_openid, message.id)

        author = getattr(message, "author", None)
        user_id = getattr(author, "member_openid", None) or getattr(author, "id", None)
        if not user_id:
            await self._reply(message, "暂时无法识别你的用户信息，请稍后重试。")
            return

        # 同一群共享上下文，所有成员的问题与 /reset 按顺序处理。
        key = message.group_openid
        lock = conversation_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            conversation_locks[key] = lock
        async with lock:
            await self._handle_question(message, question, user_id)

    async def _handle_question(self, message, question, user_id):
        group_id = message.group_openid
        remember = False
        search_context = SearchContext()
        allow_links = False
        vision = VisionInput()
        attachments = getattr(message, "attachments", None) or []

        if not question and not attachments:
            answer = "你好！请在 @我 后面写下问题，我会帮你解答。"
        elif question == "/ping":
            answer = "pong：QQ 机器人消息收发正常。"
        elif question == "/reset":
            try:
                await asyncio.to_thread(memory.clear, group_id)
                answer = "已清除当前群所有成员共享的对话记忆，我们重新开始吧！"
            except sqlite3.Error:
                log.exception("清除对话记忆失败")
                answer = "清除记忆失败，请稍后重试。"
        else:
            try:
                vision = await prepare_vision(attachments, enabled=VISION_ENABLED, model=MODEL)
            except VisionError as error:
                log.warning("图片输入未完成：%s", type(error).__name__)
                await self._reply(message, str(error))
                return
            question = question or DEFAULT_IMAGE_QUESTION
            question = question[:1500]
            allow_links = wants_links(question)
            try:
                history = await asyncio.to_thread(memory.load, group_id)
            except sqlite3.Error:
                log.exception("读取对话记忆失败")
                await self._reply(message, "暂时无法读取对话记忆，请稍后重试。")
                return
            try:
                search_context = await prepare_web_search(
                    ai, question, history, enabled=WEB_SEARCH_ENABLED, member_id=user_id,
                    image_blocks=vision.images,
                )
                response = await generate_completion(
                    ai, logger=log,
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PERSONA + GROUP_MEMORY_INSTRUCTIONS + SEARCH_ANSWER_POLICY + VISION_INSTRUCTIONS + (LINK_REQUEST_POLICY if allow_links else ""),
                        },
                        *history,
                        *search_context.messages(),
                        {"role": "user", "content": vision.user_content(user_id, question)},
                    ],
                    max_tokens=GENERATION_TOKEN_BUDGET,
                    reasoning_effort="low",
                    extra_body={"thinking": {"type": "enabled"}},
                )
                answer = (response.choices[0].message.content or "").strip()
                if (not answer and allow_links and getattr(response.choices[0], "finish_reason", None) == "stop"
                        and not getattr(response.choices[0].message, "refusal", None)):
                    answer = requested_links_fallback(search_context.sources, latest="最新" in question)
                remember = bool(answer)
                if not answer:
                    if getattr(response.choices[0], "finish_reason", None) == "length":
                        answer = "这次思考达到生成长度上限，还没有形成最终回答。请缩小问题范围后重试。"
                    else:
                        answer = "我暂时没有生成回答，请再试一次。"
            except Exception as error:
                log.error("DeepSeek API 请求失败：%s HTTP=%s", type(error).__name__, getattr(error, "status_code", None))
                answer = ("图片暂时无法识别，请确认图片清晰且格式正确，再重新发送。"
                          if vision.images and getattr(error, "status_code", None) == 400
                          else "DeepSeek 暂时不可用，请稍后重试。")

        # 普通文本消息不宜太长，避免超过 QQ 的限制。
        answer = render_search_answer(answer, search_context.sources if remember else [], limit=1600, allow_links=allow_links)
        if not answer:
            answer = "这次没有形成可用的文字回答，请换个问法再试。"
            remember = False
        if await self._reply(message, answer) and remember:
            try:
                await asyncio.to_thread(memory.append, group_id, user_id, vision.memory_question(question), answer)
            except sqlite3.Error:
                log.exception("回复已发送，但保存对话记忆失败：%s", message.id)

    async def _reply(self, message, answer):
        try:
            # GroupMessage.reply 自动关联原群消息 ID，实现被动回复。
            result = await message.reply(msg_type=0, content=answer)
            # SDK 在 HTTP 超时后返回 None，不会抛出异常。
            if result is None:
                log.warning("QQ 未返回发送确认，对话未写入记忆：%s", message.id)
                return False
            log.info("已回复消息：%s", message.id)
            return True
        except Exception:
            log.exception("向 QQ 群发送回复失败")
            return False


if __name__ == "__main__":
    configure_diagnostics(
        PROJECT_DIR / "data" / "bot.log",
        secrets=(QQ_APP_ID, QQ_APP_SECRET, DEEPSEEK_API_KEY),
    )
    # 仅接收群里 @机器人的事件（以及该 intent 下的其他公域事件，
    # 但本程序没有为那些事件编写处理函数）。
    intents = botpy.Intents(public_messages=True)
    bot = DeepSeekQQBot(intents=intents, timeout=120)
    bot.run(appid=QQ_APP_ID, secret=QQ_APP_SECRET)
