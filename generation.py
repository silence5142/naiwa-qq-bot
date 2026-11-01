"""为思考和正文保留足够预算，并记录不含正文或凭据的诊断信息。"""

import asyncio
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


GENERATION_TOKEN_BUDGET = 4096
RETRY_TOKEN_BUDGET = 8192


def configure_diagnostics(path, secrets=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    class RedactingFormatter(logging.Formatter):
        def format(self, record):
            text = super().format(record)
            for secret in secrets:
                if secret:
                    text = text.replace(secret, "[REDACTED]")
            return text

    handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    for name in ("deepseek_qq_at", "doubao_search"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return handler


def _metadata(response, logger, attempt, budget):
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    content = (getattr(message, "content", None) or "").strip()
    reason = getattr(choice, "finish_reason", None)
    refused = bool(getattr(message, "refusal", None))
    logger.info(
        "模型响应 attempt=%s budget=%s finish_reason=%s prompt_tokens=%s completion_tokens=%s reasoning_tokens=%s answer_chars=%s refused=%s",
        attempt, budget, reason, getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None), getattr(details, "reasoning_tokens", None), len(content), refused,
    )
    return content, reason, refused


async def generate_completion(ai, *, logger, total_timeout=120, **kwargs):
    async def run():
        response = await ai.chat.completions.create(**kwargs)
        budget = int(kwargs.get("max_tokens") or GENERATION_TOKEN_BUDGET)
        content, reason, refused = _metadata(response, logger, 1, budget)
        # 内容过滤/拒绝不重试；其余仅处理明确的空正文，最多两次请求。
        if not content and not refused and reason == "length" and budget < RETRY_TOKEN_BUDGET:
            logger.warning("思考耗尽生成预算且无最终正文，提升至 %s tokens 重试一次", RETRY_TOKEN_BUDGET)
            response = await ai.chat.completions.create(**{**kwargs, "max_tokens": RETRY_TOKEN_BUDGET})
            _metadata(response, logger, 2, RETRY_TOKEN_BUDGET)
        elif not content and not refused and reason == "stop":
            logger.warning("模型正常结束但没有可见正文，关闭思考重试一次")
            retry = {**kwargs, "reasoning_effort": "none",
                     "extra_body": {**(kwargs.get("extra_body") or {}), "thinking": {"type": "disabled"}}}
            response = await ai.chat.completions.create(**retry)
            _metadata(response, logger, 2, budget)
        return response

    # 两次尝试合计最多等待 120 秒，防止重试把等待时间翻倍。
    return await asyncio.wait_for(run(), timeout=total_timeout)
