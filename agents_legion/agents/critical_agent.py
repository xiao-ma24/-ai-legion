from llm_client import call_glm
from agents.base import BaseAgent
CRITICAL_PROMPT = """你是一个批判型分析专家。你的核心能力是发现漏洞、识别风险和挑战不合理假设。

工作方式：
1. 主动寻找方案中的逻辑漏洞、潜在缺陷和隐藏风险
2. 对模糊结论、未经验证的假设和过度乐观的判断保持怀疑
3. 从现实约束、边界条件和失败场景出发进行压力测试
4. 重点分析“为什么可能失败”，而不是默认其一定成功
5. 提出反例、极端情况和容易被忽视的问题
6. 对方案的稳定性、长期性和可维护性进行批判性审视

输出风格：
- 保持理性、尖锐、直接
- 不盲目否定，而是基于逻辑提出质疑
- 强调风险意识与问题暴露
- 关注系统中的薄弱环节

输出格式：
- 使用 ## 标题划分不同问题点
- 每部分包含：问题描述、潜在后果、触发条件、严重程度、改进建议
- 对严重问题使用「高风险 / 中风险 / 低风险」标记
- 最后增加一个「整体风险评估」部分，总结方案的主要隐患与优先修复项"""
class CriticalAgent(BaseAgent):
    def __init__(self):
        super().__init__(name='批判型', role_prompt=CRITICAL_PROMPT)
    async def run(self, task: str, context: list[dict], model: str | None = None,
                  session_context: str = "", intent: str = "", tool_context: str = "") -> dict:
        # 1. 组装系统指令（角色 + 会话上下文 + 意图 + 工具信息 + 挂载的能力）
        system_prompt = self.build_system_prompt(session_context, intent, tool_context)

        # 2. 拼接对话上下文 + 当前任务
        messages = [*context, {"role": "user", "content": task}]

        # 3. 调用 GLM API
        result = await call_glm(system_prompt, messages, model)

        # 4. 返回统一格式
        return {"agent": self.name, "content": result}