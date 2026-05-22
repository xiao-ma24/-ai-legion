class BaseCapability:
    """能力模板基类"""
    name: str = ""
    prompt: str = ""

    def get_prompt(self) -> str:
        return self.prompt
