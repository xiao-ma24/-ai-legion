from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """所有 Agent 的基类 — 定义统一接口"""

    def __init__(self, name: str, role_prompt: str):
        self.name = name
        self.role_prompt = role_prompt
        self._capability_prompts: list[str] = []

    def attach_capability(self, prompt: str):
        """挂载能力模板——把能力 prompt 注入到系统指令中"""
        self._capability_prompts.append(prompt)

    def build_system_prompt(self, session_context: str = "", intent: str = "", tool_context: str = "") -> str:
        """组装最终的系统指令 = 角色定位 + 会话上下文 + 当前意图 + 工具信息 + 挂载的能力模板"""
        parts = [self.role_prompt]

        if session_context:
            parts.append(f"\n## 对话上下文\n{session_context}")

        if intent:
            parts.append(f"\n## 当前交互意图\n{intent}")

        if tool_context:
            parts.append(f"\n## 外部信息（已搜索获取）\n{tool_context}")

        if self._capability_prompts:
            parts.append("\n## 能力增强\n")
            for i, cp in enumerate(self._capability_prompts, 1):
                parts.append(f"### 能力 {i}\n{cp}")

        return "\n\n".join(parts)

    @abstractmethod
    async def run(self, task: str, context: list[dict], model: str | None = None,
                  session_context: str = "", intent: str = "", tool_context: str = "") -> dict:
        """执行推理，返回 {"agent": 名字, "content": 结果}"""
        ...
