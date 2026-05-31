"""L1 总控层：复杂度分析 + 自适应调度 + 工具调用

职责：
- analyze(): 分析任务能力需求（capabilities）
- prepare_context(): 根据能力预搜索，注入外部信息
- dispatch(): 按指定角色列表创建 Agent 并行执行
- check_consistency(): analytical 路径的一致性检查
"""

import asyncio
import json
import re
from config import settings
from agents.divergent_agent import DivergentAgent
from agents.rigorous_agent import RigorousAgent
from agents.critical_agent import CriticalAgent
from capabilities import CAPABILITY_MAP
from llm_client import call_qianwen
from tools import tool_executor, TOOL_DEFINITIONS

# ── 规则快速通道 ──

SIMPLE_PATTERNS = [
    r'(翻译|translate|译成|翻译成|用.*语言.*说)',
    r'(什么是|是谁|定义|解释一下|概念|什么意思|含义)',
    r'(总结|概括|摘要|summarize|用一句话)',
    r'(多少|计算|等于|几\+几|\d+\s*[+\-×÷]\s*\d+)',
    r'(纠错|纠正|语法|错别字|拼写|修改病句)',
    r'(格式转换|格式化|转成|转换成.*格式)',
    r'(怎么读|怎么发音|拼音)',
]

# ── LLM 分析 Prompt（仅用于 new_topic） ──

ORCHESTRATOR_PROMPT = """你是一个任务分析器。你的唯一职责是分析用户输入，只输出严格 JSON。

你需要判断以下字段：

1. complexity
- 可选值："simple" / "analytical" / "strategic"
- simple：问题边界清晰、目标明确、存在标准答案、不需要多角度探索
  例：翻译、定义解释、简单问答、语法纠错、格式转换、简单计算、基础代码修复
- analytical：需要一定分析推理、存在多个合理方案、需要比较判断，但问题边界较明确
  例：学习建议、方案对比、分析报告、一般技术选型、中等代码设计
- strategic：高开放性、高不确定性、需要创造力+批判性、多目标权衡、涉及长期决策
  例：论文方向、产品战略、系统架构、竞赛选题、创业规划、长期学习路线

2. capabilities
- 可选值：["research", "writing", "coding", "product"]
- simple 任务通常只选最相关的一个
- analytical 任务选 1-2 个
- strategic 任务通常需要多个
- 选择原则：
  - research：需要资料检索、知识整理、证据分析
  - writing：需要表达组织、结构化写作、内容润色
  - coding：需要程序实现、代码设计、工程结构
  - product：需要产品分析、需求拆解、功能规划

3. budget
- "normal" 或 "high"
- simple 任务固定 normal
- analytical 任务通常 normal
- strategic 任务优先 high

输出格式严格是下面 JSON，不能添加任何额外内容：
{"complexity":"simple","capabilities":["writing"],"budget":"normal"}
"""

# ── 一致性检查 Prompt ──

CONSISTENCY_PROMPT = """你是一个一致性判断器。你需要判断两份方案的核心结论是否高度一致。

高度一致的标准：
- 核心结论相同或极为接近
- 主要推理方向一致
- 不存在明显冲突
- 可以互相补充但不矛盾

输出 JSON：
{"consistent": true, "confidence": 0.92, "better_index": 0, "reason": "两份方案核心结论一致，方案A的论述更完整"}
- consistent：true/false
- confidence：0-1 之间，你对判断的确信程度
- better_index：0=方案A更好，1=方案B更好
- reason：简短说明
"""


def _quick_classify(task: str) -> str | None:
    text = task.strip().lower()
    for pattern in SIMPLE_PATTERNS:
        if re.search(pattern, text):
            return "simple"
    return None


class Orchestrator:
    """L1 总控层"""

    def __init__(self):
        self.agent_classes = {
            "divergent": DivergentAgent,
            "rigorous": RigorousAgent,
            "critical": CriticalAgent,
        }

    # ── 分析（new_topic 时调用） ──

    async def analyze(self, task: str) -> dict:
        quick = _quick_classify(task)
        if quick:
            return {
                "complexity": "simple",
                "capabilities": ["writing"],
                "budget": "normal",
                "quick": True,
            }

        result = await call_qianwen(
            ORCHESTRATOR_PROMPT,
            [{"role": "user", "content": task}],
            model=settings.ORCHESTRATOR_MODEL,
        )
        try:
            plan = json.loads(result.strip())
            plan["quick"] = False
            return plan
        except json.JSONDecodeError:
            return {
                "complexity": "analytical",
                "capabilities": ["research", "writing"],
                "budget": "normal",
                "quick": False,
            }

    # ── 上下文准备（预搜索） ──

    async def prepare_context(
        self, task: str, capabilities: list[str],
        push_fn=None
    ) -> dict:
        """如果任务需要 research 能力，预先搜索网页和 ArXiv，返回上下文信息"""
        context_data = {"search_results": [], "arxiv_results": []}

        if "research" not in capabilities:
            return context_data

        # 网页搜索
        if push_fn:
            await push_fn("tool", "🔍 正在搜索网页...", "active")
        try:
            results = await tool_executor.execute("web_search", query=task, max_results=5)
            if results and not str(results).startswith("[工具错误"):
                context_data["search_results"] = json.loads(results) if isinstance(results, str) else results
        except Exception:
            pass
        if push_fn:
            count = len(context_data["search_results"])
            await push_fn("tool", f"🔍 网页搜索完成 — {count} 条结果", "done")

        # ArXiv 搜索
        if push_fn:
            await push_fn("tool", "📄 正在搜索学术论文 (ArXiv)...", "active")
        try:
            arxiv_r = await tool_executor.execute("arxiv_search", query=task, max_results=3)
            if arxiv_r and not str(arxiv_r).startswith("[工具错误"):
                context_data["arxiv_results"] = json.loads(arxiv_r) if isinstance(arxiv_r, str) else arxiv_r
        except Exception:
            pass
        if push_fn:
            count = len(context_data["arxiv_results"])
            await push_fn("tool", f"📄 ArXiv 论文搜索完成 — {count} 篇论文", "done")

        return context_data

    def format_context_for_agent(self, context_data: dict) -> str:
        """将搜索上下文格式化为 Agent 可读的文本"""
        parts = []
        if context_data.get("search_results"):
            parts.append("## 网页搜索结果\n")
            for i, r in enumerate(context_data["search_results"][:3], 1):
                parts.append(f"{i}. **{r['title']}**\n   {r['snippet']}\n   {r['url']}")

        if context_data.get("arxiv_results"):
            parts.append("\n## ArXiv 论文\n")
            for i, r in enumerate(context_data["arxiv_results"][:3], 1):
                authors = ", ".join(r.get("authors", []))
                parts.append(f"{i}. **{r['title']}** ({r.get('year', '')})\n   作者: {authors}\n   {r['summary'][:200]}")

        return "\n".join(parts) if parts else ""

    # ── 调度（按角色列表创建 Agent 并行执行） ──

    async def dispatch(
        self,
        task: str,
        roles: list[str],
        capabilities: list[str],
        context: list[dict],
        session_context: str = "",
        intent: str = "",
        search_context: str = "",
    ) -> list[dict]:
        agents = []
        for role_name in roles:
            agent_cls = self.agent_classes.get(role_name)
            if not agent_cls:
                continue
            agent = agent_cls()
            for cap_name in capabilities:
                cap_cls = CAPABILITY_MAP.get(cap_name)
                if cap_cls:
                    agent.attach_capability(cap_cls().get_prompt())
            agents.append(agent)

        if not agents:
            return []

        async def run_one(agent):
            return await agent.run(
                task, context,
                session_context=session_context,
                intent=intent,
                tool_context=search_context,
            )

        results = await asyncio.gather(*[run_one(a) for a in agents])
        return results

    # ── 一致性检查（analytical 路径） ──

    async def check_consistency(self, task: str, output_a: str, output_b: str) -> dict:
        user_msg = f"""## 用户任务
{task}

## 方案 A
{output_a[:2000]}

## 方案 B
{output_b[:2000]}
"""
        result = await call_qianwen(
            CONSISTENCY_PROMPT,
            [{"role": "user", "content": user_msg}],
            model=settings.ORCHESTRATOR_MODEL,
        )
        try:
            return json.loads(result.strip())
        except json.JSONDecodeError:
            return {"consistent": False, "confidence": 0, "better_index": 0}

    @staticmethod
    def complexity_label(level: str) -> str:
        return {
            "simple": "简单任务",
            "analytical": "中等任务",
            "strategic": "复杂任务",
        }.get(level, "未知")

    @staticmethod
    def intent_label(intent: str) -> str:
        return {
            "new_topic": "新话题",
            "followup": "追问深入",
            "compare_choice": "对比选择",
            "execute": "执行落地",
            "casual_confirm": "轻量反馈",
            "clarify": "澄清补充",
        }.get(intent, intent)
