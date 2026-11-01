import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from vision import MAX_IMAGE_BYTES, VisionError, VisionInput, download_qq_image, normalize_qq_image_url, prepare_vision


PNG = b"\x89PNG\r\n\x1a\nfixture"


def attachment(**changes):
    return SimpleNamespace(**dict({"content_type": "image/png", "filename": "test.png", "url": "https://gchat.qpic.cn/image", "size": len(PNG)}, **changes))


class VisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_only_does_not_download(self):
        fetcher = AsyncMock()
        result = await prepare_vision([], fetcher=fetcher)
        self.assertEqual(result.user_content("u", "hello"), "【群成员标识：u】\nhello")
        fetcher.assert_not_awaited()

    async def test_qq_image_is_downloaded_and_converted_to_inline_input(self):
        fetcher = AsyncMock(return_value=PNG)
        result = await prepare_vision([attachment(url="//gchat.qpic.cn/image")], fetcher=fetcher)
        fetcher.assert_awaited_once_with("https://gchat.qpic.cn/image")
        content = result.user_content("u", "读图")
        self.assertEqual(content[0], {"type": "text", "text": "【群成员标识：u】\n读图"})
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(content[1]["image_url"]["detail"], "high")

    async def test_more_than_three_images_fail_before_network(self):
        fetcher = AsyncMock()
        with self.assertRaisesRegex(VisionError, "最多识别 3"):
            await prepare_vision([attachment()] * 4, fetcher=fetcher)
        fetcher.assert_not_awaited()

    async def test_invalid_type_disabled_or_unsupported_model_fail_early(self):
        for changes, options in [({"content_type": "image/svg+xml"}, {}), ({}, {"enabled": False}), ({}, {"model": "deepseek-v4-pro"})]:
            fetcher = AsyncMock()
            with self.assertRaises(VisionError):
                await prepare_vision([attachment(**changes)], fetcher=fetcher, **options)
            fetcher.assert_not_awaited()

    async def test_size_guard_checks_declared_and_actual_size(self):
        fetcher = AsyncMock(return_value=PNG)
        with self.assertRaises(VisionError):
            await prepare_vision([attachment(size=MAX_IMAGE_BYTES + 1)], fetcher=fetcher)
        fetcher.assert_not_awaited()
        fetcher.return_value = b"x" * (MAX_IMAGE_BYTES + 1)
        with self.assertRaises(VisionError):
            await prepare_vision([attachment(size=None)], fetcher=fetcher)

    async def test_non_image_bytes_are_rejected(self):
        with self.assertRaises(VisionError):
            await prepare_vision([attachment()], fetcher=AsyncMock(return_value=b"<html>not image</html>"))

    async def test_network_failure_has_no_signed_url_in_user_error(self):
        signed = "https://gchat.qpic.cn/image?rkey=private-credential"
        with self.assertRaises(VisionError) as error:
            await prepare_vision([attachment(url=signed)], fetcher=AsyncMock(side_effect=RuntimeError(signed)))
        self.assertNotIn("private-credential", str(error.exception))

    async def test_original_images_are_not_in_text_memory(self):
        result = await prepare_vision([attachment()], fetcher=AsyncMock(return_value=PNG))
        text = result.memory_question("问题")
        self.assertIn("附图：1张", text)
        self.assertNotIn("data:image", text)
        self.assertNotIn("qpic.cn", text)

    async def test_failed_download_cancels_other_image_tasks(self):
        cancelled = asyncio.Event()
        async def fetch(url):
            if url.endswith("/bad"):
                await asyncio.sleep(0)
                raise VisionError("图片失效")
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        with self.assertRaises(VisionError):
            await prepare_vision([attachment(url="https://gchat.qpic.cn/bad"), attachment(url="https://gchat.qpic.cn/pending")], fetcher=fetch)
        self.assertTrue(cancelled.is_set())


class ImageUrlTests(unittest.TestCase):
    def test_only_qq_https_cdn_addresses_are_accepted(self):
        self.assertEqual(normalize_qq_image_url("gchat.qpic.cn/path"), "https://gchat.qpic.cn/path")
        for url in ["http://gchat.qpic.cn/a", "https://gchat.qpic.cn.evil.com/a", "http://127.0.0.1/a", "file:///secret", "https://u:p@gchat.qpic.cn/a", "https://gchat.qpic.cn:444/a"]:
            with self.assertRaises(VisionError):
                normalize_qq_image_url(url)


class FakeResponse:
    def __init__(self, *, status=200, headers=None, chunks=None, length=None):
        self.status = status
        self.headers = headers or {}
        self.content_length = length
        self.chunks = chunks or [PNG]
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def iter_chunked(self, size):
        for chunk in self.chunks:
            yield chunk


class DownloadTests(unittest.IsolatedAsyncioTestCase):
    def session(self, responses):
        session = SimpleNamespace(get=Mock(side_effect=responses))
        # 特殊方法从类上解析，使用独立上下文类型。
        class SessionContext:
            async def __aenter__(self):
                return session
            async def __aexit__(self, *args):
                return False
        return session, SessionContext()

    async def test_downloads_verified_bytes_without_api_authentication(self):
        session, context = self.session([FakeResponse()])
        with patch("vision.aiohttp.ClientSession", return_value=context):
            self.assertEqual(await download_qq_image("https://gchat.qpic.cn/a"), PNG)
        session.get.assert_called_once_with("https://gchat.qpic.cn/a", allow_redirects=False)

    async def test_private_redirect_is_never_followed(self):
        session, context = self.session([FakeResponse(status=302, headers={"Location": "http://127.0.0.1/private"})])
        with patch("vision.aiohttp.ClientSession", return_value=context):
            with self.assertRaises(VisionError):
                await download_qq_image("https://gchat.qpic.cn/a")
        self.assertEqual(session.get.call_count, 1)

    async def test_oversized_chunked_stream_is_stopped(self):
        session, context = self.session([FakeResponse(chunks=[PNG, b"x" * 20])])
        with patch("vision.aiohttp.ClientSession", return_value=context), patch("vision.MAX_IMAGE_BYTES", 20):
            with self.assertRaises(VisionError):
                await download_qq_image("https://gchat.qpic.cn/a")

    async def test_expired_image_has_a_clear_error(self):
        session, context = self.session([FakeResponse(status=403)])
        with patch("vision.aiohttp.ClientSession", return_value=context):
            with self.assertRaisesRegex(VisionError, "过期"):
                await download_qq_image("https://gchat.qpic.cn/a")


if __name__ == "__main__":
    unittest.main()
