"""Web 搜索工具——Bing 主力 + Baidu 备用（国内网络环境）"""

import re
import httpx


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """搜索网页，返回标题、链接、摘要"""
    results = await _search_bing(query, max_results)
    if not results:
        results = await _search_baidu(query, max_results)
    if not results:
        return [{"title": "搜索暂无结果", "url": "", "snippet": "请稍后重试或换个关键词"}]
    return results


async def _search_bing(query: str, max_results: int) -> list[dict]:
    """Bing 搜索（国内可用）"""
    try:
        async with httpx.AsyncClient(timeout=10, proxy=None, trust_env=False, follow_redirects=True) as client:
            resp = await client.get(
                "https://cn.bing.com/search",
                params={"q": query, "count": max_results},
                headers=HEADERS,
            )
            if resp.status_code != 200:
                return []
            html = resp.text
            results = []
            blocks = re.findall(
                r'<h2[^>]*>\s*<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>\s*</h2>'
                r'.*?<p[^>]*>(.*?)</p>',
                html, re.DOTALL
            )
            for url, title, snippet in blocks[:max_results]:
                clean_title = re.sub(r'<[^>]+>', '', title).strip()
                clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()[:200]
                if clean_title and '必应' not in clean_title:
                    results.append({"title": clean_title, "url": url, "snippet": clean_snippet})
            return results
    except Exception:
        return []


async def _search_baidu(query: str, max_results: int) -> list[dict]:
    """百度搜索（备用）"""
    try:
        async with httpx.AsyncClient(timeout=10, proxy=None, trust_env=False, follow_redirects=True) as client:
            resp = await client.get(
                "https://www.baidu.com/s",
                params={"wd": query, "rn": max_results, "ie": "utf-8"},
                headers=HEADERS,
            )
            if resp.status_code != 200:
                return []
            html = resp.text
            results = []
            title_blocks = re.findall(
                r'<h3[^>]*>.*?<a[^>]+href="([^"]*)"[^>]*>(.*?)</a>.*?</h3>',
                html, re.DOTALL
            )
            for url, title in title_blocks[:max_results]:
                clean_title = re.sub(r'<[^>]+>', '', title).strip()
                if clean_title:
                    results.append({"title": clean_title, "url": url, "snippet": ""})
            return results
    except Exception:
        return []


async def web_fetch(url: str) -> str:
    """获取网页文本内容"""
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url, headers=HEADERS)
            resp.raise_for_status()
            text = resp.text[:5000]
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()
    except Exception as e:
        return f"[获取失败] {e}"
