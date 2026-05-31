"""工具调用层：Agent 可操作的外部工具"""

import json
from .web_search import web_search, web_fetch
from .arxiv_search import arxiv_search


TOOL_DEFINITIONS = {
    "web_search": {
        "name": "web_search",
        "description": "搜索网页获取最新信息，返回标题、链接和摘要。参数：query（搜索关键词）, max_results（最大结果数，默认5）",
        "function": web_search,
    },
    "web_fetch": {
        "name": "web_fetch",
        "description": "获取指定网页的文本内容。参数：url（网页地址）",
        "function": web_fetch,
    },
    "arxiv_search": {
        "name": "arxiv_search",
        "description": "在 ArXiv 上搜索学术论文。参数：query（搜索关键词）, max_results（默认5）",
        "function": arxiv_search,
    },
}

TOOL_USAGE_PROMPT = """
## 可用工具

你可以使用以下工具获取外部信息。在回复中如需使用工具，请在单独一行写入：

[TOOL:工具名] 参数

示例：
[TOOL:web_search] 2025年数学建模竞赛C题分析
[TOOL:arxiv_search] large language model reasoning

工具返回的结果会自动追加到你的上下文中。
可用工具：web_search, arxiv_search, web_fetch
"""


class ToolExecutor:
    async def execute(self, name: str, **kwargs) -> str:
        tool = TOOL_DEFINITIONS.get(name)
        if not tool:
            return f"[工具错误] 未找到工具: {name}"
        try:
            result = await tool["function"](**kwargs)
            if isinstance(result, list):
                return json.dumps(result, ensure_ascii=False, indent=2)
            return str(result)
        except Exception as e:
            return f"[工具错误] {name}: {e}"

    def get_tool_prompt(self) -> str:
        return TOOL_USAGE_PROMPT


tool_executor = ToolExecutor()
