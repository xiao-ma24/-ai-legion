import asyncio
import json
import re
from config import settings
from agents.divergent_agent import DivergentAgent
from agents.rigorous_agent import RigorousAgent
from agents.critical_agent import CriticalAgent
from capabilities import CAPABILITY_MAP
from llm_client import call_qianwen

# ── 规则快速通道：命中 → 直接判 simple ──

SIMPLE_PATTERNS = [
    r'(翻译|translate|译成|翻译成|用.*语言.*说)',
    r'(什么是|是谁|定义|解释一下|概念|什么意思|含义)',
    r'(总结|概括|摘要|summarize|用一句话)',
    r'(多少|计算|等于|几\+几|\d+\s*[+\-×÷]\s*\d+)',
    r'(纠错|纠正|语法|错别字|拼写|修改病句)',
    r'(格式转换|格式化|转成|转换成.*格式)',
    r'(怎么读|怎么发音|拼音)',
]

ANALYTICAL_HINTS = [
    r'(对比|比较|哪个好|优劣|优缺点|选哪个|如何选择)',
    r'(分析|为什么|原因|因素|影响)',
    r'(建议|推荐|方法|怎么做|如何|怎么学)',
    r'(方案|设计|规划|思路)',
]


def _quick_classify(task: str) -> str | None:
    """规则快速通道：命中 simple 模式返回 'simple'，否则 None 走 LLM"""
    text = task.strip().lower()
    for pattern in SIMPLE_PATTERNS:
        if re.search(pattern, text):
            return "simple"
    return None


# ── LLM 分析 Prompt ──

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


class Orchestrator:
    """L1 总控层：复杂度估算 → 自适应调度"""

    def __init__(self):
        self.agent_classes = {
            "divergent": DivergentAgent,
            "rigorous": RigorousAgent,
            "critical": CriticalAgent,
        }

    # ── 分析 ──

    async def analyze(self, task: str) -> dict:
        """规则快判 + LLM 精确分析"""
        quick = _quick_classify(task)
        if quick:
            return {"complexity": "simple", "capabilities": ["writing"], "budget": "normal", "quick": True}

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
            return {"complexity": "analytical", "capabilities": ["research", "writing"], "budget": "normal", "quick": False}

    # ── 调度 ──

    async def dispatch(self, task: str, plan: dict, context: list[dict]) -> tuple[list[dict], str]:
        """按复杂度调度不同 Agent 组合"""
        complexity = plan.get("complexity", "analytical")
        capability_names = plan.get("capabilities", ["research"])

        if complexity == "simple":
            roles = ["rigorous"]
        elif complexity == "analytical":
            roles = ["divergent", "rigorous"]
        else:
            roles = ["divergent", "rigorous", "critical"]

        agents = []
        for role_name in roles:
            agent = self.agent_classes[role_name]()
            for cap_name in capability_names:
                cap_cls = CAPABILITY_MAP.get(cap_name)
                if cap_cls:
                    agent.attach_capability(cap_cls().get_prompt())
            agents.append(agent)

        async def run_one(agent):
            return await agent.run(task, context)

        results = await asyncio.gather(*[run_one(a) for a in agents])
        return results, complexity

    # ── 一致性检查（analytical 路径） ──

    async def check_consistency(self, task: str, output_a: str, output_b: str) -> dict:
        """判断两 Agent 输出是否高度一致"""
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

    # ── 获取复杂度对应的标签 ──

    @staticmethod
    def complexity_label(level: str) -> str:
        return {
            "simple": "简单任务",
            "analytical": "中等任务",
            "strategic": "复杂任务",
        }.get(level, "未知")
