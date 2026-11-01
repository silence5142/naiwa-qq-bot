# 奶蛙 3.2 · QQ Bot

**奶蛙 3.2** 是基于 QQ 开放平台群机器人接口和 DeepSeek 的 Python 聊天机器人，支持群聊记忆、自动联网搜索和用户附件识图。

## 功能

- 群内被 `@` 时回复；不读取普通群消息，不需要登录个人 QQ，也不依赖 NapCat。
- 每个群共享最近 **30 轮完整问答**，不同群隔离；SQLite 持久保存，重启后继续使用。
- 自动判断是否需要联网搜索，使用 DeepSeek Search API；搜索资料作为外部数据处理。
- 识别当前消息附带的图片，最多 3 张、每张 8 MiB；支持 JPG、PNG、GIF、WebP。
- 默认不附来源列表和引用编号；用户明确索要链接时保留实际检索到的网址。
- 模型空回答时受限重试，最多两次请求，共享 120 秒生成时限。
- 自定义本地人设，不需要修改源码。

联网结果和模型判断可能不完整，无法确认最新性时应明确说明。

## 准备

- [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.13.6 或更新版本；默认使用 Python 3.11，也支持 Python 3.12。uv 可自动下载所需 Python。
- QQ 开放平台创建的群机器人，已配置群聊能力及测试群，持有 AppID 和 AppSecret。
- DeepSeek API Key、可用额度和支持本项目请求格式的模型。

本项目默认模型名为 `deepseek-flash`，识图模块对该模型及其兼容别名做检查。请先确认账户支持相应模型、思考、图像理解和搜索接口；更换其他提供商或任意模型不能保证直接兼容。

## 使用 uv 快速开始

Windows 可在 PowerShell 中安装 uv：

```powershell
winget install --id=astral-sh.uv -e
```

安装完成后重新打开 PowerShell，执行 `uv --version` 确认安装成功。其他系统和安装方式参见 [uv 安装文档](https://docs.astral.sh/uv/getting-started/installation/)。

在仓库根目录安装依赖并创建配置：

```powershell
uv sync --locked
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
if (-not (Test-Path persona.txt)) { Copy-Item persona.example.txt persona.txt }
notepad .env
```

填写 `.env` 中的三项凭据。再打开 `persona.txt` 修改人设；本地人设和凭据均已加入 Git 忽略规则。若没有 `persona.txt`，程序使用 `persona.example.txt`。

启动：

```powershell
uv run --locked python bot.py
```

窗口显示机器人连接成功后，在测试群发送 `@机器人 /ping`。关闭时在运行窗口按 **Ctrl+C**。修改配置或人设后，关闭再启动即可生效。不要同时运行多个使用同一账号的实例。

如果需要让 DeepSeek 请求绕过本机代理，可以在本次 PowerShell 窗口先执行以下命令，再启动：

```powershell
$env:NO_PROXY = "$env:NO_PROXY,api.deepseek.com"
```

Linux / macOS 同样使用 `uv sync --locked` 安装依赖、`uv run --locked python bot.py` 启动；首次配置可执行 `test -f .env || cp .env.example .env` 和 `test -f persona.txt || cp persona.example.txt persona.txt`。

uv 自动管理项目下的 `.venv`，无需手动激活虚拟环境。首次运行可能需要联网下载 Python 和依赖。使用 `--locked` 可以检查锁文件是否与依赖声明一致，避免运行时意外更新锁文件。

### 使用 pip 安装

已安装 Python 3.11 或 3.12 的用户也可以使用 pip。在项目目录执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

按上文创建 `.env` 和 `persona.txt` 后，运行 `.\.venv\Scripts\python.exe bot.py`。Linux / macOS 对应的 Python 路径为 `.venv/bin/python`。

## 配置和人设

| 配置 | 默认值 / 用途 |
| --- | --- |
| `QQ_APP_ID` | 必填，QQ 机器人 AppID |
| `QQ_APP_SECRET` | 必填，QQ 机器人 AppSecret |
| `DEEPSEEK_API_KEY` | 必填，DeepSeek API Key |
| `DEEPSEEK_MODEL` | `deepseek-flash` |
| `WEB_SEARCH_ENABLED` | `true`，是否自动搜索 |
| `VISION_ENABLED` | `true`，是否读取附图 |
| `MEMORY_DB_PATH` | `data/memory.sqlite3`，相对路径以项目目录为基准 |

`persona.txt` 为 UTF-8 文本，只填写希望机器人的语气、身份和回答习惯。程序还会附加群成员区分、检索资料处理和识图规则。群记忆会影响回答，换人设后可在群内使用 `/reset` 清除旧上下文。

代码默认限制：输入 1500 字符、发送回复 1600 字符；最终模型使用 low 思考、4096 tokens 生成预算（含思考与正文）。若因长度耗尽而没有正文，最多提升至 8192 tokens 重试一次；正常结束但正文为空时，最多关闭思考重试一次。过滤或明确拒绝不触发这些重试。QQ HTTP 超时为 120 秒。

自动检索最多两条查询、保留三个结果。纯文字搜索阶段最多 35 秒，带图最多 60 秒；生成和图片下载另有时限，因此总回复耗时可能长于单独的生成时限。搜索及模型调用可能产生费用。

## 群内命令

| 消息 | 行为 |
| --- | --- |
| `@机器人 /ping` | 检查 QQ 收发，不请求模型 |
| `@机器人 /reset` | 清除当前群所有成员共享的记忆 |
| `@机器人 你好` | 普通对话 |
| `@机器人 帮我解释这张图`，并附图 | 分析当轮图片 |
| `@机器人 给我相关网页链接` | 允许回答中保留实际相关网址 |

当前 **任何能 @ 机器人的群成员均可使用 `/reset`**，没有管理员权限控制。群成员会共享上下文，请在部署前向群成员说明。

## 数据存储

- `data/memory.sqlite3` 保存群标识、成员标识、问题和实际发送成功的回答；失败或未确认发送的回答不记入历史。
- 原图以内存数据传给模型，不写入聊天数据库；历史只保存附图数量标记和文字解读，需要重新核对图片细节时应重发图片。
- `data/bot.log` 保存生成状态、用量等诊断信息，自动轮转并脱敏配置凭据；SDK 也可能生成自己的日志。日志可能包含群标识和消息标识，请勿公开。
- 对话、当轮图片及检索信息会发送给相应模型服务，QQ 消息收发通过 QQ 平台完成。

`.env`、本地人设、数据库、日志、虚拟环境均不属于开源发布内容。请仅分享模板配置，不要上传运行目录的完整备份。

## 验证和排查

离线测试，不连接 QQ、不调用 DeepSeek：

```powershell
uv run --locked python -m unittest discover -s tests -q
```

单独验证 DeepSeek（会调用真实 API，可能收费）：

```powershell
uv run --locked python test_deepseek.py
```

- 缺少 `botpy`：运行 `uv sync --locked` 安装依赖，并通过 `uv run --locked python bot.py` 启动；pip 用户确认使用 `.venv` 中的 Python。
- 找不到 `uv`：安装后重新打开终端，执行 `uv --version`；也可参考 uv 文档中的其他安装方式。
- QQ 鉴权失败或群内无响应：核对凭据、平台权限、测试群与平台要求的网络设置；本程序接收 `GROUP_AT_MESSAGE_CREATE` 事件。
- `/ping` 正常、AI 回复失败：检查模型权限、API 额度与网络；查看本地诊断日志。
- 提示没有生成回答：查看 `finish_reason`、token 用量和正文字数；本版已加入受限重试，但不能保证每次都能得到正文。
- 搜索返回失败：确认账号支持搜索接口；可暂时设置 `WEB_SEARCH_ENABLED=false`。
- 识图失败：检查模型是否兼容、图片类型和大小，以及 QQ 图片链接是否失效。
- QQ 发送超时：检查网络、回复有效期和平台频率限制。未收到发送确认不代表成功，本程序不会把它写入记忆。

## 依赖

`pyproject.toml` 声明项目依赖和 Python 版本范围，`uv.lock` 锁定直接与间接依赖，`.python-version` 指定默认 Python 3.11。`requirements.txt` 由锁文件导出，供 pip 用户使用；`.venv` 不提交到 Git。

添加或更新依赖后，重新导出 pip 清单并运行测试：

```powershell
uv export --locked --format requirements-txt --no-dev --no-emit-project --no-header --no-annotate --output-file requirements.txt
uv run --locked python -m unittest discover -s tests -q
```

添加依赖使用 `uv add 包名`，删除依赖使用 `uv remove 包名`；修改 `pyproject.toml` 后执行 `uv lock` 更新锁文件。将 `pyproject.toml`、`uv.lock` 和导出的 `requirements.txt` 一同提交。

QQ 接口使用 `qq-botpy-v2` 社区兼容包（`import botpy`），它并非腾讯原版 SDK 的官方发布包。请勿与同名导入的 `qq-botpy` 同时安装。

- [QQ 开放平台](https://q.qq.com/)
- [QQ botpy 上游](https://github.com/tencent-connect/botpy)
- [qq-botpy-v2 包](https://pypi.org/project/qq-botpy-v2/)
- [DeepSeek 文档](https://api-docs.deepseek.com/zh-cn/)

## 许可证

项目源码按 [MIT License](LICENSE) 发布；第三方依赖保留其各自许可证。本项目不附带任何平台账号、密钥或个人数据。
