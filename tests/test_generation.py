import asyncio
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from generation import configure_diagnostics, generate_completion


def response(content, reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, reasoning_content="private reasoning text"), finish_reason=reason)],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=800, completion_tokens_details=SimpleNamespace(reasoning_tokens=800)),
    )


class GenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.create = AsyncMock(return_value=response("完成"))
        self.ai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        self.logger = Mock()

    async def test_normal_answer_does_not_retry_or_log_reasoning_text(self):
        r = await generate_completion(self.ai, logger=self.logger, max_tokens=4096)
        self.assertEqual(r.choices[0].message.content, "完成")
        self.create.assert_awaited_once()
        self.assertNotIn("private reasoning text", repr(self.logger.mock_calls))

    async def test_empty_length_response_retries_with_more_budget(self):
        self.create.side_effect = [response(None, "length"), response("最终答案")]
        r = await generate_completion(self.ai, logger=self.logger, model="deepseek-flash", max_tokens=4096, reasoning_effort="low")
        self.assertEqual(r.choices[0].message.content, "最终答案")
        self.assertEqual(self.create.await_count, 2)
        self.assertEqual(self.create.call_args_list[0].kwargs["max_tokens"], 4096)
        self.assertEqual(self.create.call_args_list[1].kwargs["max_tokens"], 8192)
        self.assertEqual(self.create.call_args_list[1].kwargs["reasoning_effort"], "low")

    async def test_repeated_length_exhaustion_is_bounded(self):
        self.create.return_value = response("", "length")
        r = await generate_completion(self.ai, logger=self.logger, max_tokens=4096)
        self.assertEqual(r.choices[0].finish_reason, "length")
        self.assertEqual(self.create.await_count, 2)

    async def test_filtered_response_is_not_retried(self):
        self.create.return_value = response(None, "content_filter")
        await generate_completion(self.ai, logger=self.logger, max_tokens=4096)
        self.create.assert_awaited_once()

    async def test_partial_visible_answer_is_not_retried(self):
        self.create.return_value = response("部分正文", "length")
        await generate_completion(self.ai, logger=self.logger, max_tokens=4096)
        self.create.assert_awaited_once()

    async def test_empty_stop_response_retries_once_with_thinking_disabled(self):
        self.create.side_effect = [response("", "stop"), response("https://example.org/video")]
        result = await generate_completion(self.ai, logger=self.logger, max_tokens=4096, reasoning_effort="low",
                                            extra_body={"thinking": {"type": "enabled"}, "other": "preserved"})
        self.assertEqual(result.choices[0].message.content, "https://example.org/video")
        self.assertEqual(self.create.await_count, 2)
        self.assertEqual(self.create.call_args.kwargs["max_tokens"], 4096)
        self.assertEqual(self.create.call_args.kwargs["reasoning_effort"], "none")
        self.assertEqual(self.create.call_args.kwargs["extra_body"], {"thinking": {"type": "disabled"}, "other": "preserved"})

    async def test_two_empty_stop_responses_are_bounded(self):
        self.create.return_value = response("", "stop")
        await generate_completion(self.ai, logger=self.logger, max_tokens=4096)
        self.assertEqual(self.create.await_count, 2)

    async def test_refusal_is_not_retried_even_with_empty_stop_content(self):
        declined = response(None, "stop")
        declined.choices[0].message.refusal = "model refusal"
        self.create.return_value = declined
        await generate_completion(self.ai, logger=self.logger, max_tokens=4096)
        self.create.assert_awaited_once()

    async def test_retry_does_not_reset_the_total_deadline(self):
        async def blocked(**kwargs):
            if self.create.await_count == 1:
                return response("", "stop")
            await asyncio.Event().wait()
        self.create.side_effect = blocked
        with self.assertRaises(asyncio.TimeoutError):
            await generate_completion(self.ai, logger=self.logger, total_timeout=.02, max_tokens=4096)
        self.assertEqual(self.create.await_count, 2)

    async def test_higher_budget_is_not_retried_without_bound(self):
        self.create.return_value = response(None, "length")
        await generate_completion(self.ai, logger=self.logger, max_tokens=8192)
        self.create.assert_awaited_once()

    async def test_entire_generation_obeys_one_deadline(self):
        async def blocked(**kwargs):
            await asyncio.Event().wait()
        self.create.side_effect = blocked
        with self.assertRaises(asyncio.TimeoutError):
            await generate_completion(self.ai, logger=self.logger, total_timeout=0.01, max_tokens=4096)


class DiagnosticTests(unittest.TestCase):
    def test_saved_log_redacts_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bot.log"
            handler = configure_diagnostics(path, secrets=["secret-test-key"])
            logger = logging.getLogger("deepseek_qq_at")
            try:
                logger.info("test credential=%s", "secret-test-key")
                handler.flush()
                text = path.read_text(encoding="utf-8")
                self.assertIn("[REDACTED]", text)
                self.assertNotIn("secret-test-key", text)
            finally:
                for name in ["deepseek_qq_at", "doubao_search"]:
                    logging.getLogger(name).removeHandler(handler)
                handler.close()


if __name__ == "__main__":
    unittest.main()
