"""单独测试 DeepSeek API（不连接 QQ）。"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
key = os.getenv("DEEPSEEK_API_KEY", "").strip()
if not key:
    raise RuntimeError("请先配置 .env 中的 DEEPSEEK_API_KEY")

client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=120, max_retries=0)
response = client.chat.completions.create(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
    messages=[{"role": "user", "content": "请回复：DeepSeek API 测试成功"}],
    max_tokens=80,
    extra_body={"thinking": {"type": "disabled"}},
)
print((response.choices[0].message.content or "").strip())
