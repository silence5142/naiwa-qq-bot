import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from web_search import SearchContext, prepare_web_search, render_search_answer, requested_links_fallback, safe_source_url, wants_links


def tool_call(query):
    return SimpleNamespace(function=SimpleNamespace(name="web_search", arguments=json.dumps({"query": query})))


def page(url="https://example.com/news", title="官方资料", **extra):
    return {"title": title, "url": url, "summary": "可核实的摘要", "date": "2026-10-30", **extra}


class AutoSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.message = SimpleNamespace(tool_calls=None)
        self.create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=self.message)]))
        self.post = AsyncMock(return_value={"results": [page()]})
        self.ai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)), post=self.post)

    async def test_disabled_search_makes_no_planning_or_search_requests(self):
        result = await prepare_web_search(self.ai, "你好", [], enabled=False)
        self.assertEqual(result.status, "disabled")
        self.assertIn('"状态": "disabled"', result.messages()[0]["content"])
        self.create.assert_not_awaited()
        self.post.assert_not_awaited()

    async def test_regular_chat_does_not_call_search_endpoint(self):
        result = await prepare_web_search(self.ai, "你好", [])
        self.assertEqual(result.status, "not_needed")
        self.post.assert_not_awaited()

    async def test_search_planner_receives_images_when_present(self):
        images = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fixture"}}]
        await prepare_web_search(self.ai, "查图里型号的最新价格", [], image_blocks=images)
        request = self.create.call_args.kwargs
        self.assertEqual(request["messages"][-1]["content"][1], images[0])
        self.assertEqual(request["timeout"], 30)

    async def test_model_selected_search_uses_real_endpoint_and_bounded_results(self):
        self.message.tool_calls = [tool_call("今日新闻")]
        result = await prepare_web_search(self.ai, "今天有什么新闻", [])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.sources[0]["url"], "https://example.com/news")
        self.assertEqual(self.post.call_args.args[0], "/v1/tools/search")
        self.assertEqual(self.post.call_args.kwargs["body"], {"query": "今日新闻", "count": 3, "need_content": True})
        self.assertEqual(result.messages()[0]["role"], "user")
        self.assertIn('"number": 1', result.messages()[0]["content"])

    async def test_at_most_two_searches_and_both_topics_get_sources(self):
        self.message.tool_calls = [tool_call("问题一"), tool_call("问题二"), tool_call("忽略第三个")]
        self.post.side_effect = [
            {"results": [page("https://example.com/a"), page("https://example.com/b")]},
            {"results": [page("https://example.org/c"), page("https://example.com/a")]},
        ]
        result = await prepare_web_search(self.ai, "两个最新问题", [])
        self.assertEqual(self.post.await_count, 2)
        self.assertEqual([s["url"] for s in result.sources], ["https://example.com/a", "https://example.org/c", "https://example.com/b"])

    async def test_duplicate_queries_only_search_once(self):
        self.message.tool_calls = [tool_call("同一个问题"), tool_call("同一个问题")]
        await prepare_web_search(self.ai, "最新情况", [])
        self.assertEqual(self.post.await_count, 1)

    async def test_same_page_tracking_and_fragment_variants_are_deduplicated(self):
        self.message.tool_calls = [tool_call("查询")]
        self.post.return_value = {"results": [page("https://example.com/news#one"), page("https://example.com/news?utm_source=test"), page("https://example.com/other")]}
        result = await prepare_web_search(self.ai, "最新情况", [])
        self.assertEqual(len(result.sources), 2)

    async def test_invalid_tool_arguments_fail_without_executing_anything(self):
        self.message.tool_calls = [SimpleNamespace(function=SimpleNamespace(name="web_search", arguments="not-json"))]
        with self.assertLogs("doubao_search", level="WARNING"):
            result = await prepare_web_search(self.ai, "问题", [])
        self.assertEqual(result.status, "failed")
        self.post.assert_not_awaited()

    async def test_failed_search_is_not_presented_as_success(self):
        self.message.tool_calls = [tool_call("今日新闻")]
        self.post.side_effect = RuntimeError("simulated service failure")
        result = await prepare_web_search(self.ai, "问题", [])
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.sources, [])
        self.assertIn('"状态": "failed"', result.messages()[0]["content"])

    async def test_empty_results_are_identified(self):
        self.message.tool_calls = [tool_call("找不到的资料")]
        self.post.return_value = {"results": []}
        result = await prepare_web_search(self.ai, "问题", [])
        self.assertEqual(result.status, "empty")
        self.assertEqual(result.sources, [])

    async def test_partial_failure_preserves_successful_sources(self):
        self.message.tool_calls = [tool_call("问题一"), tool_call("问题二")]
        self.post.side_effect = [RuntimeError("unavailable"), {"results": [page()]}]
        result = await prepare_web_search(self.ai, "问题", [])
        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.sources), 1)

    async def test_malformed_provider_payload_is_not_success(self):
        self.message.tool_calls = [tool_call("查询")]
        self.post.return_value = {"unexpected": []}
        result = await prepare_web_search(self.ai, "问题", [])
        self.assertEqual(result.status, "failed")

    async def test_member_identifiers_are_removed_from_web_query(self):
        self.message.tool_calls = [tool_call("历史-member-id 当前-member-id 北京天气")]
        history = [{"role": "user", "content": "【群成员标识：历史-member-id】\n我住北京"}]
        await prepare_web_search(self.ai, "天气怎么样", history, member_id="当前-member-id")
        query = self.post.call_args.kwargs["body"]["query"]
        self.assertNotIn("member-id", query)
        self.assertIn("北京天气", query)

    async def test_key_like_queries_are_not_sent_to_search(self):
        self.message.tool_calls = [tool_call("搜索 sk-exampleSecret123456")]
        with self.assertLogs("doubao_search", level="WARNING"):
            result = await prepare_web_search(self.ai, "问题", [])
        self.assertEqual(result.status, "failed")
        self.post.assert_not_awaited()

    async def test_private_or_invalid_source_links_are_discarded(self):
        self.message.tool_calls = [tool_call("资料")]
        self.post.return_value = {"results": [page("http://127.0.0.1/secret"), page("file:///secret"), page()]}
        result = await prepare_web_search(self.ai, "问题", [])
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0]["url"], "https://example.com/news")


class SourceRenderingTests(unittest.TestCase):
    def test_explicit_link_requests_are_distinguished_from_no_source_links(self):
        for question in ["能转发泛式b站最新视频的链接吗", "给我官网地址", "下载地址是什么", "把URL发我", "不要来源链接，只给我视频链接"]:
            self.assertTrue(wants_links(question), question)
        for question in ["解释一下这道题", "不要附来源链接", "不用给我链接", "讲清楚，不用加链接"]:
            self.assertFalse(wants_links(question), question)

    def test_requested_urls_survive_but_citation_numbers_do_not(self):
        url = "https://example.com/news"
        for answer in [url, "这里：" + url + "[1]", "来源：\n[1] " + url, "[1](" + url + ")"]:
            rendered = render_search_answer(answer, [page()], allow_links=True)
            self.assertIn(url, rendered)
            self.assertNotIn("[1]", rendered)
        self.assertEqual(render_search_answer("这里：" + url + "[1]", [page()]), "这里：")

    def test_requested_link_fallback_only_uses_safe_actual_results_and_does_not_claim_latest(self):
        rendered = requested_links_fallback([page(), page("http://127.0.0.1/private")], latest=True)
        self.assertIn("https://example.com/news", rendered)
        self.assertIn("无法确认", rendered)
        self.assertNotIn("127.0.0.1", rendered)
        self.assertEqual(requested_links_fallback([]), "")
    def test_long_answer_is_limited_without_a_source_footer(self):
        sources = [page("https://example.com/" + "a" * 250)]
        rendered = render_search_answer("回答" * 1500, sources)
        self.assertLessEqual(len(rendered), 1600)
        self.assertTrue(rendered.endswith("…"))
        self.assertNotIn(sources[0]["url"], rendered)
        self.assertNotIn("来源：", rendered)

    def test_citations_and_known_source_links_are_removed(self):
        sources = [page()]
        for text, expected in [
            ("答案。[1][2][3]", "答案。"),
            ("参考[官方资料](https://example.com/news)。", "参考官方资料。"),
            ("答案。\n\n来源：\n[1] 官方资料\nhttps://example.com/news", "答案。"),
            ("答案。\n<https://example.com/news>", "答案。"),
        ]:
            self.assertEqual(render_search_answer(text, sources), expected)

    def test_old_history_sources_are_removed_without_a_current_search(self):
        text = "上次说的结论。[2][3]\n\n来源:\n[2] 旧网页\nhttps://example.org/old"
        self.assertEqual(render_search_answer(text, []), "上次说的结论。")

    def test_functional_links_and_code_are_preserved(self):
        text = "下载：https://download.example.org/app\nitems[1]\n`[1]`\n```python\nvalues = [1]\n```"
        self.assertEqual(render_search_answer(text, [page()]), text)

    def test_empty_body_and_zero_limit_return_empty_text(self):
        self.assertEqual(render_search_answer("来源：\n[1] https://example.com/news", []), "")
        self.assertEqual(render_search_answer("内容", [], limit=0), "")

    def test_no_search_has_no_invented_footer(self):
        self.assertEqual(render_search_answer("你好", []), "你好")
        self.assertEqual(SearchContext().messages(), [])

    def test_credential_bearing_or_non_public_links_are_rejected(self):
        for url in ["https://user:password@example.com/", "http://localhost/a", "http://10.0.0.1/a", "javascript:alert(1)"]:
            self.assertIsNone(safe_source_url(url))


if __name__ == "__main__":
    unittest.main()
