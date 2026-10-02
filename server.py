#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dsh-search-bridge —— 自托管多引擎搜索网关

用途：作为 DSH（DeepSeek Harness）内置「官方搜索」provider 的自建后端。
DSH 侧只需把 web-search-deepseek 的 baseURL 指向本服务，即可用上
"模型改写查询 + 多引擎检索" 的搜索能力，全程零外部 API 费用。

接口：
  POST /v1/messages   Anthropic Messages 兼容层（DSH 官方搜索 provider 对接）
  GET  /search        通用 JSON 搜索： /search?q=关键词&engines=bing,sogou&limit=10
  GET  /healthz       健康检查
  GET  /              说明页

引擎：bing / sogou / baidu（内置直抓，免 key，国内直连）
可选：mojeek（免 key，境外）；brave（需 BRAVE_API_KEY）

大脑（可选）：调用本地大模型改写查询、多轮检索（BRAIN_* 环境变量控制）。
未配置或调用失败时自动降级为「直接检索原始查询」。

依赖：Python 3.9+ 标准库，无第三方包。
"""

from __future__ import annotations

import html as html_mod
import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

VERSION = "1.0.0"
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def env_bool(name: str, default: bool) -> bool:
    raw = env(name, "")
    if raw == "":
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    try:
        return int(env(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env(name, str(default)))
    except ValueError:
        return default


CFG = {
    "host": env("LISTEN_HOST", "0.0.0.0"),
    "port": env_int("LISTEN_PORT", 8090),
    # 密钥：逗号分隔，多个都接受。为空且未允许匿名时，所有请求 401。
    "keys": [k.strip() for k in env("SEARCH_BRIDGE_KEYS", "").split(",") if k.strip()],
    "allow_anon": env_bool("SEARCH_BRIDGE_ALLOW_ANON", False),
    # 检索
    # 顺序即优先级：中文场景 sogou/baidu 明显优于 bing（bing 中文分词差）
    "engines": [e.strip().lower() for e in env("SEARCH_ENGINES", "sogou,baidu,bing").split(",") if e.strip()],
    "limit": env_int("SEARCH_LIMIT", 10),
    "engine_timeout": env_float("SEARCH_TIMEOUT", 12.0),
    "resolve_links": env_bool("RESOLVE_LINKS", True),
    "merge_dedupe": True,
    # 单次 /v1/messages 的总预算（DSH 侧搜索超时是 60s）
    "budget": env_float("SEARCH_BUDGET", 50.0),
    # 大脑（本地模型，可选）
    "brain_enabled": env_bool("BRAIN_ENABLED", True),
    "brain_base": env("BRAIN_BASE_URL", "http://127.0.0.1:8888/v1").rstrip("/"),
    "brain_key": env("BRAIN_API_KEY", ""),
    "brain_model": env("BRAIN_MODEL", "deepseek-v4.1-flash"),
    "brain_rounds": env_int("BRAIN_MAX_ROUNDS", 3),
    "brain_timeout": env_float("BRAIN_TIMEOUT", 25.0),
    "brain_per_query": env_int("BRAIN_RESULTS_PER_QUERY", 6),
    # 可选境外引擎
    "brave_key": env("BRAVE_API_KEY", ""),
}

LOG_LOCK = threading.Lock()


def log(msg: str) -> None:
    with LOG_LOCK:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# HTTP 抓取工具
# --------------------------------------------------------------------------


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


_OPENER = build_opener()
_NO_REDIRECT_OPENER = build_opener(_NoRedirect)


def fetch(url: str, timeout: float, *, headers: dict | None = None, no_redirect: bool = False):
    """返回 (status, final_url, body_text)。失败返回 (0, url, '')。"""
    hdrs = {
        "User-Agent": DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    if headers:
        hdrs.update(headers)
    opener = _NO_REDIRECT_OPENER if no_redirect else _OPENER
    try:
        with opener.open(Request(url, headers=hdrs), timeout=timeout) as resp:
            raw = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
            try:
                text = raw.decode(charset, errors="replace")
            except LookupError:
                text = raw.decode("utf-8", errors="replace")
            return resp.status, resp.geturl(), text
    except HTTPError as exc:
        try:
            raw = exc.read()
            text = raw.decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover
            text = ""
        return exc.code, getattr(exc, "url", url), text
    except (URLError, OSError, ValueError, TimeoutError) as exc:
        log(f"fetch failed: {url[:120]} :: {exc}")
        return 0, url, ""


def fetch_json(url: str, payload: dict, timeout: float, *, headers: dict | None = None):
    body = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "User-Agent": DEFAULT_UA}
    if headers:
        hdrs.update(headers)
    try:
        with _OPENER.open(Request(url, data=body, headers=hdrs, method="POST"), timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", errors="replace"))
    except HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8", errors="replace"))
        except Exception:
            return exc.code, None
    except (URLError, OSError, ValueError, TimeoutError) as exc:
        log(f"fetch_json failed: {url} :: {exc}")
        return 0, None


# --------------------------------------------------------------------------
# HTML 清洗与跳转链接解析
# --------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def clean_text(fragment: str) -> str:
    if not fragment:
        return ""
    text = _COMMENT_RE.sub(" ", fragment)
    text = _TAG_RE.sub("", text)
    text = html_mod.unescape(text)
    return _WS_RE.sub(" ", text).strip()


_LINK_CACHE: dict[str, str] = {}
_LINK_LOCK = threading.Lock()


_JS_REDIRECT = re.compile(r"""window\.location\.replace\(\s*["']([^"']+)["']""")
_META_REDIRECT = re.compile(r"""content=["'][^"']*URL=['"]?([^"';,\s]+)""", re.I)


def _probe_redirect(url: str, timeout: float) -> str:
    """返回 URL 指向的真实地址：302 Location、JS 跳转或 meta refresh；拿不到返回 ''。"""
    body = ""
    try:
        with _NO_REDIRECT_OPENER.open(Request(url, headers={"User-Agent": DEFAULT_UA}), timeout=timeout) as resp:
            loc = resp.headers.get("Location", "") or ""
            if loc:
                return loc
            body = resp.read(4000).decode("utf-8", errors="replace")
    except HTTPError as exc:
        loc = exc.headers.get("Location", "") or ""
        if loc:
            return loc
        try:
            body = exc.read(4000).decode("utf-8", errors="replace")
        except Exception:
            return ""
    except Exception:
        return ""
    match = _JS_REDIRECT.search(body) or _META_REDIRECT.search(body)
    return match.group(1) if match else ""


def resolve_real_url(link: str, timeout: float = 6.0) -> str:
    """把百度/搜狗的跳转链接解析成真实 URL；失败则原样返回。"""
    if not link or not CFG["resolve_links"]:
        return link
    parsed = urlparse(link)
    if not parsed.netloc or "link?" not in link:
        return link
    with _LINK_LOCK:
        cached = _LINK_CACHE.get(link)
    if cached:
        return cached

    real = link
    current = link
    for _ in range(3):  # 最多跟三跳（含 JS 跳转页）
        loc = _probe_redirect(current, timeout)
        if not loc:
            break
        loc = urljoin(current, loc)
        if not urlparse(loc).netloc:
            break
        if urlparse(loc).netloc != parsed.netloc:
            real = loc
            break
        current = loc
    if real == link:
        _, final_url, _ = fetch(link, timeout)
        if final_url and urlparse(final_url).netloc and urlparse(final_url).netloc != parsed.netloc:
            real = final_url

    with _LINK_LOCK:
        if len(_LINK_CACHE) > 4000:
            _LINK_CACHE.clear()
        _LINK_CACHE[link] = real
    return real


def _fallback_snippet(window_html: str, title: str) -> str:
    """从结果块 HTML 里捞一段像人话的文本当摘要（搜索引擎结构常变，做兜底）。"""
    if not window_html:
        return ""
    window_html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", window_html, flags=re.S | re.I)
    for frag in re.findall(r">([^<>]{25,400})<", window_html):
        text = clean_text(frag)
        if len(text) < 25 or text in title:
            continue
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        if cjk >= 8 or (cjk >= 3 and len(text) > 60):
            return text[:600]
    return ""


# --------------------------------------------------------------------------
# 各搜索引擎解析
# --------------------------------------------------------------------------

_BING_BLOCK = re.compile(r'<li class="b_algo".*?(?=<li class="b_algo"|</ol>)', re.S)
_BING_TITLE = re.compile(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_BING_SNIPPET = re.compile(r'<p[^>]*>(.*?)</p>', re.S)

_SOGOU_TITLE = re.compile(
    r'<h3[^>]*class="[^"]*vr-title[^"]*"[^>]*>\s*(?:<!--.*?-->\s*)*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.S,
)
_SOGOU_SNIPPET = re.compile(r'class="[^"]*(?:space-txt|star-wiki)[^"]*"[^>]*>(.*?)</(?:p|div|span)>', re.S)

_BAIDU_TITLE = re.compile(
    r'<h3[^>]*>\s*(?:<[^>]+>\s*)*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.S,
)
_BAIDU_SNIPPET = re.compile(
    r'class="[^"]*(?:summary-text|c-abstract|content-right)[^"]*"[^>]*>(.*?)</(?:span|div)>',
    re.S,
)

_MOJEEK_BLOCK = re.compile(r'<li class="web-result.*?</li>', re.S)
_MOJEEL_TITLE = re.compile(r'<a[^>]*class="ob"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)


def _mk(engine: str, title: str, url: str, snippet: str = "") -> dict | None:
    title, url, snippet = clean_text(title), (url or "").strip(), clean_text(snippet)
    if not url or not title or not url.lower().startswith(("http://", "https://")):
        return None
    return {"engine": engine, "title": title[:200], "url": url, "snippet": snippet[:600]}


def search_bing(query: str, limit: int) -> list[dict]:
    url = f"https://cn.bing.com/search?q={quote(query)}&count={max(limit + 4, 12)}&ensearch=0"
    status, _, body = fetch(url, CFG["engine_timeout"])
    if not body:
        return []
    out: list[dict] = []
    for block in _BING_BLOCK.findall(body):
        mt = _BING_TITLE.search(block)
        if not mt:
            continue
        ms = _BING_SNIPPET.search(block)
        item = _mk("bing", mt.group(2), mt.group(1), ms.group(1) if ms else "")
        if item:
            out.append(item)
        if len(out) >= limit:
            break
    return out


def search_sogou(query: str, limit: int) -> list[dict]:
    base = "https://www.sogou.com"
    url = f"{base}/web?query={quote(query)}"
    status, _, body = fetch(url, CFG["engine_timeout"])
    if not body:
        return []
    out: list[dict] = []
    seen_links: set[str] = set()
    for mt in _SOGOU_TITLE.finditer(body):
        href, title = urljoin(base, mt.group(1)), mt.group(2)
        if href in seen_links:
            continue
        seen_links.add(href)
        window = body[mt.end(): mt.end() + 1500]
        ms = _SOGOU_SNIPPET.search(window)
        snippet = ms.group(1) if ms else _fallback_snippet(window, clean_text(title))
        item = _mk("sogou", title, href, snippet)
        if item:
            out.append(item)
        if len(out) >= limit:
            break
    return out


def search_baidu(query: str, limit: int) -> list[dict]:
    base = "https://www.baidu.com"
    url = f"{base}/s?wd={quote(query)}&rn={max(limit + 4, 12)}"
    status, _, body = fetch(url, CFG["engine_timeout"], headers={"Referer": "https://www.baidu.com/"})
    if not body:
        return []
    out: list[dict] = []
    seen_links: set[str] = set()
    for mt in _BAIDU_TITLE.finditer(body):
        href, title = urljoin(base, mt.group(1)), mt.group(2)
        if href in seen_links:
            continue
        seen_links.add(href)
        window = body[mt.end(): mt.end() + 2500]
        ms = _BAIDU_SNIPPET.search(window)
        snippet = ms.group(1) if ms else _fallback_snippet(window, clean_text(title))
        item = _mk("baidu", title, href, snippet)
        if item:
            out.append(item)
        if len(out) >= limit:
            break
    return out


def search_mojeek(query: str, limit: int) -> list[dict]:
    url = f"https://www.mojeek.com/search?q={quote(query)}"
    status, _, body = fetch(url, CFG["engine_timeout"])
    if not body:
        return []
    out: list[dict] = []
    for block in _MOJEEK_BLOCK.findall(body):
        mt = _MOJEEL_TITLE.search(block)
        if not mt:
            continue
        item = _mk("mojeek", mt.group(2), mt.group(1))
        if item:
            out.append(item)
        if len(out) >= limit:
            break
    return out


def search_brave(query: str, limit: int) -> list[dict]:
    if not CFG["brave_key"]:
        return []
    url = f"https://api.search.brave.com/res/v1/web/search?q={quote(query)}&count={limit}"
    status, _, body = fetch(url, CFG["engine_timeout"], headers={"X-Subscription-Token": CFG["brave_key"], "Accept": "application/json"})
    if not body:
        return []
    try:
        data = json.loads(body)
    except ValueError:
        return []
    out: list[dict] = []
    for row in (data.get("web", {}) or {}).get("results", []) or []:
        item = _mk("brave", row.get("title", ""), row.get("url", ""), row.get("description", ""))
        if item:
            out.append(item)
        if len(out) >= limit:
            break
    return out


ENGINES = {
    "bing": search_bing,
    "sogou": search_sogou,
    "baidu": search_baidu,
    "mojeek": search_mojeek,
    "brave": search_brave,
}


# --------------------------------------------------------------------------
# 检索编排
# --------------------------------------------------------------------------


def _normalize_key(item: dict) -> str:
    parsed = urlparse(item["url"])
    path = parsed.path.rstrip("/")
    return f"{parsed.netloc.lower()}{path}"


def _query_terms(query: str) -> list[str]:
    """提取用于相关性判断的词（长度 >= 2）。"""
    return [t for t in re.split(r"[\s,，。、;；:：/|]+", query) if len(t) >= 2]


def _relevant(item: dict, terms: list[str]) -> bool:
    if not terms:
        return True
    haystack = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
    return any(term.lower() in haystack for term in terms)


def run_search(query: str, engines: list[str] | None = None, limit: int | None = None) -> list[dict]:
    """并发跑多引擎，按引擎优先级合并去重，返回 [{engine,title,url,snippet}]。"""
    query = (query or "").strip()
    if not query:
        return []
    limit = limit or CFG["limit"]
    chosen = [e for e in (engines or CFG["engines"]) if e in ENGINES]
    if not chosen:
        chosen = ["sogou", "baidu", "bing"]
    # 纯英文查询让 bing 打头（搜狗/百度英文能力弱）；中文查询保持配置顺序
    if not any("\u4e00" <= ch <= "\u9fff" for ch in query):
        chosen.sort(key=lambda name: 0 if name == "bing" else 1)
    priority = {name: idx for idx, name in enumerate(chosen)}

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(6, len(chosen))) as pool:
        futures = {pool.submit(ENGINES[name], query, limit): name for name in chosen}
        for fut in as_completed(futures, timeout=CFG["engine_timeout"] + 8):
            name = futures[fut]
            try:
                part = fut.result() or []
            except Exception as exc:  # pragma: no cover
                log(f"engine {name} error: {exc}")
                part = []
            log(f"engine {name}: {len(part)} hits for {query!r}")
            for idx, item in enumerate(part):
                item["_rank"] = idx
                results.append(item)

    if CFG["resolve_links"] and results:
        with ThreadPoolExecutor(max_workers=8) as pool:
            resolved = list(pool.map(lambda r: resolve_real_url(r["url"]), results))
        for item, real in zip(results, resolved):
            if real:
                item["url"] = real

    results.sort(key=lambda r: (priority.get(r["engine"], 99), r.get("_rank", 99)))

    # 相关性过滤：与查询词完全无交集的结果（典型是 bing 中文分词跑偏）先剔除；
    # 过滤后太少就保留原列表，避免"全军覆没"。
    terms = _query_terms(query)
    if terms:
        filtered = [r for r in results if _relevant(r, terms)]
        if len(filtered) >= 3:
            results = filtered

    merged: list[dict] = []
    index: dict[str, int] = {}
    for item in results:
        key = _normalize_key(item)
        pos = index.get(key)
        if pos is not None:
            if not merged[pos].get("snippet") and item.get("snippet"):
                merged[pos]["snippet"] = item["snippet"]
            continue
        index[key] = len(merged)
        item.pop("_rank", None)
        merged.append(item)
    return merged[: max(limit * 2, 12)]


# --------------------------------------------------------------------------
# 大脑：用本地模型改写查询 / 多轮检索（可选）
# --------------------------------------------------------------------------

_BRAIN_SYS = (
    "你是一个搜索规划助手。用户给出一个检索意图，你的任务是调用 web_search 工具完成检索。"
    "如果第一次结果不理想，可以换关键词再搜一次（最多 {rounds} 次）。"
    "不要自己编造答案，只负责决定搜索词。"
)

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "在互联网上搜索给定关键词，返回标题、链接和摘要",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索关键词，中文优先"}},
            "required": ["query"],
        },
    },
}


def brain_available() -> bool:
    return bool(CFG["brain_enabled"] and CFG["brain_base"])


def _brain_chat(messages: list[dict], timeout: float):
    payload = {
        "model": CFG["brain_model"],
        "messages": messages,
        "tools": [_SEARCH_TOOL],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 700,
    }
    headers = {}
    if CFG["brain_key"]:
        headers["Authorization"] = f"Bearer {CFG['brain_key']}"
    status, data = fetch_json(f"{CFG['brain_base']}/chat/completions", payload, timeout, headers=headers)
    if status != 200 or not isinstance(data, dict):
        log(f"brain call failed (status={status})")
        return None
    try:
        return data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None


def agentic_search(query: str, deadline: float) -> list[dict]:
    """用本地模型规划检索词并多轮搜索；任何异常都退化为直接检索。"""
    if not brain_available():
        return run_search(query)

    collected: list[dict] = []
    messages = [
        {"role": "system", "content": _BRAIN_SYS.format(rounds=CFG["brain_rounds"])},
        {"role": "user", "content": query},
    ]
    rounds = max(1, CFG["brain_rounds"])
    for round_no in range(rounds):
        if time.monotonic() > deadline:
            break
        msg = _brain_chat(messages, min(CFG["brain_timeout"], max(2.0, deadline - time.monotonic())))
        if msg is None:
            break
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        messages.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": calls,
        })
        for call in calls:
            raw_args = (call.get("function") or {}).get("arguments") or "{}"
            try:
                sub_query = (json.loads(raw_args).get("query") or query).strip()
            except ValueError:
                sub_query = query
            found = run_search(sub_query, limit=CFG["brain_per_query"])
            collected.extend(found)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", "call_0"),
                "content": json.dumps(
                    [{"title": r["title"], "url": r["url"], "snippet": r["snippet"][:200]} for r in found[:6]],
                    ensure_ascii=False,
                )[:4000],
            })
        if time.monotonic() > deadline:
            break

    if not collected:
        return run_search(query)
    merged: list[dict] = []
    seen: set[str] = set()
    for item in collected:
        key = _normalize_key(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged[: max(CFG["limit"] * 2, 12)]


# --------------------------------------------------------------------------
# Anthropic 兼容响应
# --------------------------------------------------------------------------


def build_anthropic_response(query: str, sources: list[dict]) -> dict:
    items = []
    citations = []
    for src in sources:
        entry = {"type": "web_search_result", "url": src["url"], "title": src["title"]}
        items.append(entry)
        if src.get("snippet"):
            citations.append({
                "type": "web_search_result_location",
                "url": src["url"],
                "title": src["title"],
                "cited_text": src["snippet"],
            })
    content: list[dict] = [{
        "type": "web_search_tool_result",
        "tool_use_id": f"srvtoolu_{uuid.uuid4().hex[:24]}",
        "content": items,
    }]
    if citations:
        engines = ",".join(sorted({s.get("engine", "?") for s in sources}))
        content.append({
            "type": "text",
            "text": f"检索到 {len(items)} 条结果（引擎：{engines}）。",
            "citations": citations,
        })
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": "dsh-search-bridge",
        "content": content,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": max(1, len(query) // 2), "output_tokens": 0},
    }


# --------------------------------------------------------------------------
# HTTP 服务
# --------------------------------------------------------------------------


def check_auth(handler: BaseHTTPRequestHandler) -> bool:
    if CFG["allow_anon"] and not CFG["keys"]:
        return True
    supplied = handler.headers.get("x-api-key", "").strip()
    if not supplied:
        auth = handler.headers.get("authorization", "").strip()
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
    if not CFG["keys"]:
        return True  # 没配密钥且非强制
    return supplied in CFG["keys"]


def extract_query(body: dict) -> str:
    parts: list[str] = []
    for msg in body.get("messages") or []:
        if (msg.get("role") or "") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
    text = " ".join(parts).strip()
    text = re.sub(r"^Perform a web search for the query:\s*", "", text, flags=re.I)
    return text.strip()


class Handler(BaseHTTPRequestHandler):
    server_version = f"dsh-search-bridge/{VERSION}"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # 用统一日志
        log(f"{self.address_string()} {fmt % args}")

    # -- helpers ----------------------------------------------------------
    def _send(self, status: int, payload: dict | str, ctype: str = "application/json; charset=utf-8") -> None:
        raw = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except BrokenPipeError:  # pragma: no cover
            pass

    def _read_json(self) -> dict:
        length = int(self.headers.get("content-length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            return {}

    # -- routes -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/healthz", "/health"):
            return self._send(200, {"ok": True, "version": VERSION})
        if parsed.path == "/" and not parse_qs(parsed.query).get("q"):
            return self._send(200, {
                "service": "dsh-search-bridge",
                "version": VERSION,
                "engines": [e for e in CFG["engines"] if e in ENGINES],
                "endpoints": ["POST /v1/messages", "GET /search?q=...&engines=bing,sogou&limit=10", "GET /healthz"],
                "brain": {"enabled": brain_available(), "base": CFG["brain_base"], "model": CFG["brain_model"]},
            })
        if parsed.path not in ("/search", "/"):
            return self._send(404, {"error": "not found"})
        if not check_auth(self):
            return self._send(401, {"error": "unauthorized: missing or invalid api key"})
        params = parse_qs(parsed.query)
        query = (params.get("q") or [""])[0].strip()
        if not query:
            return self._send(400, {"error": "missing q"})
        engines = [e.strip().lower() for e in (params.get("engines") or [""])[0].split(",") if e.strip()] or None
        limit = int((params.get("limit") or [str(CFG["limit"])])[0] or CFG["limit"])
        started = time.monotonic()
        results = run_search(query, engines=engines, limit=limit)
        return self._send(200, {
            "query": query,
            "engines": engines or CFG["engines"],
            "count": len(results),
            "elapsed": round(time.monotonic() - started, 2),
            "results": results,
        })

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in ("/v1/messages", "/messages"):
            return self._send(404, {"error": "not found"})
        if not check_auth(self):
            return self._send(401, {"error": {"type": "authentication_error", "message": "invalid api key"}})
        body = self._read_json()
        query = extract_query(body)
        if not query:
            return self._send(400, {"error": {"type": "invalid_request_error", "message": "no user query found"}})
        deadline = time.monotonic() + CFG["budget"]
        started = time.monotonic()
        try:
            sources = agentic_search(query, deadline)
        except Exception as exc:  # pragma: no cover
            log(f"agentic_search failed: {exc}; falling back to direct")
            sources = run_search(query)
        response = build_anthropic_response(query, sources)
        log(f"messages: query={query!r} sources={len(sources)} elapsed={time.monotonic() - started:.2f}s")
        return self._send(200, response)


def main() -> None:
    if not CFG["keys"] and not CFG["allow_anon"]:
        log("WARNING: SEARCH_BRIDGE_KEYS 未设置，所有请求将返回 401（可设 SEARCH_BRIDGE_ALLOW_ANON=1 放行）")
    log(f"dsh-search-bridge {VERSION} listening on {CFG['host']}:{CFG['port']}")
    log(f"engines={CFG['engines']} brain={'on' if brain_available() else 'off'} base={CFG['brain_base']}")
    httpd = ThreadingHTTPServer((CFG["host"], CFG["port"]), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        log("bye")
        httpd.server_close()


if __name__ == "__main__":
    sys.exit(main())