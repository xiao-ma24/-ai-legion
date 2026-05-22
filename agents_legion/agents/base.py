from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """所有 Agent 的基类 — 定义统一接口"""

    def __init__(self, name: str, role_prompt: str):
        self.name = name
        self.role_prompt = role_prompt   # 角色定位 prompt，定义了这个 Agent 的身份、职责、行为准则等
        self._capability_prompts: list[str] = [] # 能力模板列表，后续可以通过 attach_capability() 方法挂载能力模板

    def attach_capability(self, prompt: str):
        """挂载能力模板——把能力 prompt 注入到系统指令中"""
        self._capability_prompts.append(prompt)

    def build_system_prompt(self) -> str:
        """组装最终的系统指令 = 角色定位 + 挂载的能力模板"""
        parts = [self.role_prompt]
        if self._capability_prompts:
            parts.append("\n## 能力增强\n")
            for i, cp in enumerate(self._capability_prompts, 1):
                parts.append(f"### 能力 {i}\n{cp}")
        return "\n\n".join(parts)

    @abstractmethod
    async def run(self, task: str, context: list[dict], model: str | None = None) -> dict:
        """执行推理，返回 {"agent": 名字, "content": 结果}"""
        ...
