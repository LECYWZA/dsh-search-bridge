# dsh-search-bridge

自托管的多引擎搜索网关。作为 DSH（DeepSeek Harness）**内置"官方搜索"provider 的自建后端**使用：
DSH 侧只改配置（不改插件、不加 MCP），搜索就由本服务完成——多引擎真实检索 + 本地大模型改写查询/多轮追问，
全程零外部 API 费用。

## 它提供什么

| 接口 | 说明 |
|---|---|
| `POST /v1/messages` | Anthropic Messages 兼容层。实现服务端 `web_search` 工具语义，返回 `web_search_tool_result` 结果块，DSH 的 `dsh-web-search-deepseek` provider 可直接指向它 |
| `GET /search?q=…&engines=bing,sogou&limit=10` | 通用 JSON 搜索接口（便于调试、也可给别的程序用） |
| `GET /healthz` | 健康检查（不需要密钥） |

检索引擎：`sogou`、`baidu`、`bing`（内置直抓，免 key、国内直连），可选 `mojeek`、`brave`（需 `BRAVE_API_KEY`）。
跳转链接会自动解析成真实地址；同名结果会去重、按引擎优先级与相关性排序。

"大脑"：可选调用本地大模型（默认走 OpenWrt 上的 `http://192.168.1.1:8888/v1` → Spark sglang）把自然语言意图
拆解成检索词并可多轮搜索，最终结果由服务自己汇总 —— 这就是"官方搜索"的机制，只是模型跑在你自己机器上。

## 部署（OpenWrt / 任何有 Docker 的机器）

```bash
mkdir -p /data/dsh-search-bridge && cd /data/dsh-search-bridge
# 把 server.py Dockerfile docker-compose.yml 放到这个目录
cp .env.example .env && vi .env        # 至少改 SEARCH_BRIDGE_KEYS
docker compose up -d --build
curl -s http://127.0.0.1:8090/healthz
```

不用 compose 也可以：

```bash
docker build -t dsh-search-bridge:1.0.0 .
docker run -d --name dsh-search-bridge --restart unless-stopped --network host \
  -e SEARCH_BRIDGE_KEYS=你的密钥 \
  -e BRAIN_BASE_URL=http://192.168.1.1:8888/v1 -e BRAIN_API_KEY=luojiecong \
  dsh-search-bridge:1.0.0
```

## 接到 DSH 的"官方搜索"

在 `$DSH_HOME/profiles/web/cordis.patch.yml` 追加（它会覆盖 searxng 插件对 `web` 行的覆盖）：

```yaml
- id: web
  config:
    searchProvider: deepseek-official   # 用回内置的官方搜索 provider
    fetchProvider: http

- id: web-search-deepseek
  config:
    apiKey: 你的密钥                     # 与 SEARCH_BRIDGE_KEYS 一致
    baseURL: http://127.0.0.1:8090/v1    # 指向本服务
    model: dsh-search-bridge             # 由本服务响应决定，写什么都行
```

改完重启 dsh 服务生效（或在 GUI：设置 → 插件 → 插件配置 → Web search → Endpoint 里改）。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `LISTEN_PORT` | `8090` | 监听端口 |
| `SEARCH_BRIDGE_KEYS` | 空 | 允许的密钥，逗号分隔；为空且未开匿名时所有请求 401 |
| `SEARCH_BRIDGE_ALLOW_ANON` | `0` | `1` = 不校验密钥 |
| `SEARCH_ENGINES` | `sogou,baidu,bing` | 引擎与优先级 |
| `SEARCH_LIMIT` | `10` | 单次结果条数上限 |
| `SEARCH_TIMEOUT` | `12` | 单引擎请求超时（秒） |
| `RESOLVE_LINKS` | `1` | 是否把跳转链接解析成真实 URL |
| `SEARCH_BUDGET` | `50` | 单次 `/v1/messages` 总预算（秒，DSH 侧搜索超时 60s） |
| `BRAIN_ENABLED` | `1` | 是否启用本地模型做查询规划 |
| `BRAIN_BASE_URL` | `http://127.0.0.1:8888/v1` | 本地模型端点（OpenAI 兼容） |
| `BRAIN_API_KEY` | 空 | 本地模型密钥 |
| `BRAIN_MODEL` | `deepseek-v4.1-flash` | 模型名 |
| `BRAIN_MAX_ROUNDS` | `3` | 最多几轮检索 |
| `BRAIN_TIMEOUT` | `25` | 单次模型调用超时（秒） |
| `BRAVE_API_KEY` | 空 | 可选，启用 Brave 引擎 |

## 说明与限制

- 引擎是抓取搜索结果页实现的：百度/搜狗/Bing 对中文/英文各有强弱（Bing 中文分词差，所以中文默认不让它打头），
  页面结构变化时需要更新解析规则（`server.py` 顶部的正则区）。
- 纯标准库实现，无需第三方依赖；镜像很小，离线也能构建。
- 每轮"大脑"调用会占用本地模型算力；不想要可设 `BRAIN_ENABLED=0`，退化为纯多引擎检索。