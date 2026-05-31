from agents.base import BaseAgent
from llm_client import call_deepseek
# ═══════════════════════════════════════════
#Rigorous Agent 的角色定位 Prompt — 定义了"严谨型"的思维方式
# ═══════════════════════════════════════════
RIGOROUS_PROMPT = """你是一个严谨型推理专家。你的核心能力是逻辑分析、结构化推理和可行性验证。

工作方式：
1. 对问题进行系统性拆解，明确核心目标与关键约束条件
2. 基于事实、逻辑和因果关系逐步推导，而不是凭直觉下结论
3. 对每个方案进行可行性、合理性和一致性验证
4. 优先考虑最稳健、最可执行、最符合现实条件的路径
5. 主动识别推理漏洞、隐含假设和逻辑跳跃
6. 在多个方案之间进行权衡分析，给出清晰的优先级判断

输出风格：
- 负责逻辑推理与可行性验证
- 强调因果链
- 强调现实约束
- 强调执行步骤
- 输出必须结构严谨
- 保持客观、冷静、结构化
- 避免情绪化表达和过度发散
- 强调“为什么成立”
- 每一步推理都尽量可解释

输出格式：
- 使用 ## 标题划分不同分析部分
- 每部分包含：核心判断、推理依据、关键约束、可行性分析、潜在风险
- 最后增加一个「最终结论」部分，明确推荐方案及原因"""
class RigorousAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="严谨型", role_prompt=RIGOROUS_PROMPT)
    async def run(self, task: str, context: list[dict], model: str | None = None,
                  session_context: str = "", intent: str = "", tool_context: str = "") -> dict:
        # 1. 组装系统指令（角色 + 会话上下文 + 意图 + 工具信息 + 挂载的能力）
        system_prompt = self.build_system_prompt(session_context, intent, tool_context)

        # 2. 拼接对话上下文 + 当前任务
        messages = [*context, {"role": "user", "content": task}]

        # 3. 调用 DeepSeek API
        result = await call_deepseek(system_prompt, messages, model)

        # 4. 返回统一格式
        return {"agent": self.name, "content": result}
