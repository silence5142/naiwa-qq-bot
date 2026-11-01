import asyncio
import importlib
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from memory import ConversationMemory
from web_search import SearchContext
from vision import DEFAULT_IMAGE_QUESTION, VisionInput


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "memory.sqlite3"
        self.memory = ConversationMemory(self.path)

    def test_keeps_thirty_complete_turns_in_order_on_disk(self):
        for number in range(32):
            self.memory.append("group", f"user-{number % 2}", f"question-{number}", f"answer-{number}")
        history = self.memory.load("group")
        self.assertEqual(len(history), 60)
        self.assertEqual(history[0], {"role": "user", "content": "【群成员标识：user-0】\nquestion-2"})
        self.assertEqual(history[-1], {"role": "assistant", "content": "answer-31"})
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM turns").fetchone()[0], 30)

    def test_reopening_database_preserves_history_and_reset(self):
        self.memory.append("group", "user", "remember me", "remembered")
        reopened = ConversationMemory(self.path)
        self.assertEqual(reopened.load("group"), self.memory.load("group"))
        reopened.clear("group")
        self.assertEqual(ConversationMemory(self.path).load("group"), [])

    def test_reset_clears_all_members_in_selected_group_only(self):
        for group, user in [("a", "one"), ("a", "two"), ("b", "one")]:
            self.memory.append(group, user, "question", "answer")
        self.assertEqual(len(self.memory.load("a")), 4)
        self.memory.clear("a")
        self.assertEqual(self.memory.load("a"), [])
        self.assertEqual(len(self.memory.load("b")), 2)


class BotMemoryTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # 导入主程序时也只使用临时数据库和假密钥，不访问真实账户。
        cls.import_folder = tempfile.TemporaryDirectory()
        with patch.dict(os.environ, {
            "QQ_APP_ID": "test-app",
            "QQ_APP_SECRET": "test-secret",
            "DEEPSEEK_API_KEY": "test-key",
            "MEMORY_DB_PATH": str(Path(cls.import_folder.name) / "import.sqlite3"),
        }):
            cls.module = importlib.import_module("bot")

    @classmethod
    def tearDownClass(cls):
        asyncio.run(cls.module.ai.close())
        cls.import_folder.cleanup()

    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.memory = ConversationMemory(Path(self.folder.name) / "test.sqlite3")
        self.create = AsyncMock(return_value=self.response("reply"))
        self.fake_ai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        for name, value in [("memory", self.memory), ("ai", self.fake_ai)]:
            patcher = patch.object(self.module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(self.module, "prepare_web_search", AsyncMock(return_value=SearchContext()))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.module.recent_message_ids.clear()
        self.module.conversation_locks.clear()
        self.bot = self.module.DeepSeekQQBot.__new__(self.module.DeepSeekQQBot)
        self.number = 0
        # 压低常规测试日志；错误分支仍由 assertLogs 明确捕获。
        patcher = patch.object(self.module.log, "info")
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def response(text):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    def message(self, content, group="group", user="user", attachments=None):
        self.number += 1
        return self.module.GroupMessage(
            api=SimpleNamespace(post_group_message=AsyncMock()),
            event_id="event",
            data={"id": f"message-{self.number}", "content": content,
                  "group_openid": group, "author": {"member_openid": user}, "attachments": attachments or []},
        )

    async def send(self, message):
        await self.bot.on_group_at_message_create(message)

    async def test_followup_includes_prior_delivered_answer(self):
        await self.send(self.message("my name is Ming"))
        await self.send(self.message("what is my name?"))
        messages = self.create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1:], [
            {"role": "user", "content": "【群成员标识：user】\nmy name is Ming"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "【群成员标识：user】\nwhat is my name?"},
        ])

    async def test_members_share_history_but_other_groups_are_isolated(self):
        await self.send(self.message("private", group="a", user="one"))
        await self.send(self.message("hello", group="a", user="two"))
        messages = self.create.call_args.kwargs["messages"]
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["content"], "【群成员标识：one】\nprivate")
        self.assertEqual(messages[-1]["content"], "【群成员标识：two】\nhello")
        await self.send(self.message("hello", group="b", user="one"))
        self.assertEqual(len(self.create.call_args.kwargs["messages"]), 2)

    async def test_reset_skips_ai_and_next_question_has_no_history(self):
        await self.send(self.message("remember this"))
        reset = self.message(" /reset ", user="another-member")
        await self.send(reset)
        self.assertEqual(self.create.await_count, 1)
        self.assertEqual(self.memory.load("group"), [])
        reset._api.post_group_message.assert_awaited_once()
        await self.send(self.message("new conversation"))
        self.assertEqual(len(self.create.call_args.kwargs["messages"]), 2)

    async def test_commands_empty_answers_and_errors_are_not_remembered(self):
        await self.send(self.message("/ping"))
        await self.send(self.message("  "))
        self.create.assert_not_awaited()
        self.module.prepare_web_search.assert_not_awaited()
        self.create.return_value = self.response("   ")
        await self.send(self.message("empty response"))
        self.create.side_effect = RuntimeError("simulated API failure")
        with self.assertLogs(self.module.log, level="ERROR"):
            await self.send(self.message("failed request"))
        self.assertEqual(self.memory.load("group"), [])

    async def test_failed_qq_reply_is_not_remembered(self):
        message = self.message("hello")
        message._api.post_group_message.side_effect = RuntimeError("simulated send failure")
        with self.assertLogs(self.module.log, level="ERROR"):
            await self.send(message)
        self.assertEqual(self.memory.load("group"), [])

    async def test_duplicate_event_does_not_call_ai_twice(self):
        message = self.message("hello")
        await self.send(message)
        await self.send(message)
        self.assertEqual(self.create.await_count, 1)
        self.assertEqual(len(self.memory.load("group")), 2)

    async def test_qq_timeout_without_exception_is_not_remembered(self):
        message = self.message("hello")
        message._api.post_group_message.return_value = None
        with self.assertLogs(self.module.log, level="WARNING"):
            await self.send(message)
        self.assertEqual(self.memory.load("group"), [])

    async def test_only_sent_content_is_saved_with_length_limits(self):
        self.create.return_value = self.response("a" * 2000)
        await self.send(self.message("q" * 2000))
        history = self.memory.load("group")
        self.assertEqual(len(history[0]["content"].split("\n", 1)[1]), 1500)
        self.assertEqual(len(history[1]["content"]), 1600)

    async def test_missing_author_does_not_share_anonymous_memory(self):
        message = self.message("hello", user=None)
        await self.send(message)
        self.create.assert_not_awaited()
        message._api.post_group_message.assert_awaited_once()

    async def test_reset_waits_for_inflight_reply_then_clears_it(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def blocked_response(**kwargs):
            started.set()
            await release.wait()
            return self.response("reply")

        self.create.side_effect = blocked_response
        first = asyncio.create_task(self.send(self.message("remember me")))
        await asyncio.wait_for(started.wait(), timeout=3)
        reset = asyncio.create_task(self.send(self.message("/reset", user="another-member")))
        await asyncio.sleep(0)
        self.assertFalse(reset.done())
        release.set()
        await asyncio.wait_for(asyncio.gather(first, reset), timeout=3)
        self.assertEqual(self.memory.load("group"), [])

    async def test_consecutive_messages_wait_and_receive_updated_history(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def blocked_response(**kwargs):
            started.set()
            await release.wait()
            return self.response("reply")

        self.create.side_effect = blocked_response
        first = asyncio.create_task(self.send(self.message("first")))
        await asyncio.wait_for(started.wait(), timeout=3)
        second = asyncio.create_task(self.send(self.message("second", user="another-member")))
        await asyncio.sleep(0)
        self.assertEqual(self.create.await_count, 1)
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=3)
        self.assertEqual(self.create.call_args.kwargs["messages"][1]["content"], "【群成员标识：user】\nfirst")
        self.assertEqual(len(self.memory.load("group")), 4)

    async def test_other_groups_can_reply_during_a_blocked_group_request(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def selective_response(**kwargs):
            if kwargs["messages"][-1]["content"].endswith("\nfirst"):
                started.set()
                await release.wait()
            return self.response("reply")

        self.create.side_effect = selective_response
        first = asyncio.create_task(self.send(self.message("first", group="a")))
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            await asyncio.wait_for(self.send(self.message("second", group="b")), timeout=3)
            self.assertFalse(first.done())
        finally:
            release.set()
            await asyncio.wait_for(first, timeout=3)
        self.assertEqual(len(self.memory.load("a")), 2)
        self.assertEqual(len(self.memory.load("b")), 2)

    async def test_search_sources_reach_model_and_are_saved_as_delivered(self):
        source = {"title": "官方资料", "url": "https://example.com/news", "text": "核实后的资料", "date": "2026-10-30"}
        self.module.prepare_web_search.return_value = SearchContext("ok", [source])
        self.create.return_value = self.response("根据最新资料，情况如下。[1]")
        message = self.message("查一下最新情况")
        await self.send(message)
        messages = self.create.call_args.kwargs["messages"]
        self.assertIn("本轮联网检索状态", messages[-2]["content"])
        self.assertEqual(messages[-1]["content"], "【群成员标识：user】\n查一下最新情况")
        sent = message._api.post_group_message.call_args.kwargs["content"]
        self.assertNotIn(source["url"], sent)
        self.assertNotIn("[1]", sent)
        self.assertEqual(sent, "根据最新资料，情况如下。")
        self.assertEqual(self.memory.load("group")[-1]["content"], sent)

    async def test_old_source_footer_is_removed_from_followup_reply(self):
        self.memory.append("group", "user", "上个问题", "旧回答。[1]\n\n来源：\n[1] 旧网页\nhttps://example.org/old")
        self.create.return_value = self.response("还是之前的结论。[1]\n\n来源：\n[1] 旧网页\nhttps://example.org/old")
        message = self.message("再说一下")
        await self.send(message)
        sent = message._api.post_group_message.call_args.kwargs["content"]
        self.assertEqual(sent, "还是之前的结论。")
        self.assertEqual(self.memory.load("group")[-1]["content"], sent)

    async def test_requested_video_link_is_delivered_and_remembered(self):
        url = "https://www.bilibili.com/video/BV1GJ411x7h7/"
        self.module.prepare_web_search.return_value = SearchContext("ok", [{"title": "相关视频", "url": url}])
        self.create.return_value = self.response("可以看这个：" + url)
        message = self.message("能转发泛式b站最新视频的链接吗")
        await self.send(message)
        sent = message._api.post_group_message.call_args.kwargs["content"]
        self.assertIn(url, sent)
        self.assertNotIn("来源：", sent)
        self.assertIn("本轮用户明确索要", self.create.call_args.kwargs["messages"][0]["content"])
        self.assertEqual(self.memory.load("group")[-1]["content"], sent)

    async def test_empty_stop_retry_can_deliver_requested_link(self):
        empty = self.response("");empty.choices[0].finish_reason = "stop"
        url = "https://www.bilibili.com/video/BV1GJ411x7h7/"
        self.create.side_effect = [empty, self.response(url)]
        message = self.message("给我视频链接")
        with self.assertLogs(self.module.log, level="WARNING"):
            await self.send(message)
        self.assertEqual(self.create.await_count, 2)
        self.assertEqual(self.create.call_args.kwargs["reasoning_effort"], "none")
        self.assertEqual(message._api.post_group_message.call_args.kwargs["content"], url)
        self.assertEqual(len(self.memory.load("group")), 2)

    async def test_two_empty_answers_use_actual_requested_links_without_claiming_latest(self):
        empty = self.response("");empty.choices[0].finish_reason = "stop"
        self.create.return_value = empty
        url = "https://www.bilibili.com/video/BV1GJ411x7h7/"
        self.module.prepare_web_search.return_value = SearchContext("ok", [{"title": "相关页面", "url": url}])
        message = self.message("能转发泛式b站最新视频的链接吗")
        with self.assertLogs(self.module.log, level="WARNING"):
            await self.send(message)
        self.assertEqual(self.create.await_count, 2)
        sent = message._api.post_group_message.call_args.kwargs["content"]
        self.assertIn(url, sent)
        self.assertIn("无法确认", sent)
        self.assertNotIn("我暂时没有生成回答", sent)

    async def test_filtered_link_request_does_not_retry_or_use_source_fallback(self):
        empty = self.response("");empty.choices[0].finish_reason = "content_filter"
        self.create.return_value = empty
        url = "https://example.org/filtered"
        self.module.prepare_web_search.return_value = SearchContext("ok", [{"title": "资料", "url": url}])
        message = self.message("给我链接")
        await self.send(message)
        self.create.assert_awaited_once()
        self.assertNotIn(url, message._api.post_group_message.call_args.kwargs["content"])
        self.assertEqual(self.memory.load("group"), [])

    async def test_source_only_answer_is_not_sent_empty_or_saved(self):
        self.create.return_value = self.response("来源：\n[1] https://example.org/old")
        message = self.message("查一下")
        await self.send(message)
        sent = message._api.post_group_message.call_args.kwargs["content"]
        self.assertTrue(sent)
        self.assertIn("没有形成可用", sent)
        self.assertEqual(self.memory.load("group"), [])

    async def test_search_failure_status_reaches_answer_model(self):
        self.module.prepare_web_search.return_value = SearchContext("failed")
        await self.send(self.message("查今天的新闻"))
        messages = self.create.call_args.kwargs["messages"]
        self.assertIn('"状态": "failed"', messages[-2]["content"])

    async def test_empty_reasoning_budget_retry_only_remembers_final_answer(self):
        empty = self.response("")
        empty.choices[0].finish_reason = "length"
        self.create.side_effect = [empty, self.response("最终答案")]
        with self.assertLogs(self.module.log, level="WARNING"):
            await self.send(self.message("较复杂的问题"))
        self.assertEqual(self.create.await_count, 2)
        history = self.memory.load("group")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1]["content"], "最终答案")

    async def test_no_final_answer_after_retry_does_not_write_history(self):
        empty = self.response("")
        empty.choices[0].finish_reason = "length"
        self.create.return_value = empty
        message = self.message("较复杂的问题")
        with self.assertLogs(self.module.log, level="WARNING"):
            await self.send(message)
        self.assertEqual(self.create.await_count, 2)
        self.assertEqual(self.memory.load("group"), [])
        self.assertIn("生成长度上限", message._api.post_group_message.call_args.kwargs["content"])

    async def test_image_only_message_has_default_question_and_real_image_input(self):
        images = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fixture", "detail": "high"}}]
        message = self.message("", attachments=[{"content_type": "image/png", "url": "https://gchat.qpic.cn/a"}])
        with patch.object(self.module, "prepare_vision", AsyncMock(return_value=VisionInput(images))):
            await self.send(message)
        content = self.create.call_args.kwargs["messages"][-1]["content"]
        self.assertIn(DEFAULT_IMAGE_QUESTION, content[0]["text"])
        self.assertEqual(content[1], images[0])
        self.assertEqual(self.module.prepare_web_search.call_args.kwargs["image_blocks"], images)
        saved = self.memory.load("group")
        self.assertIn("历史未保存原图", saved[0]["content"])
        self.assertNotIn("base64", saved[0]["content"])

    async def test_unsupported_attachment_skips_search_generation_and_memory(self):
        message = self.message("看看这个", attachments=[{"content_type": "application/pdf", "url": "https://gchat.qpic.cn/a"}])
        with self.assertLogs(self.module.log, level="WARNING"):
            await self.send(message)
        self.create.assert_not_awaited()
        self.module.prepare_web_search.assert_not_awaited()
        self.assertEqual(self.memory.load("group"), [])


if __name__ == "__main__":
    unittest.main()
