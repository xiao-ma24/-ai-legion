from decouple import config


class Settings:
    DEEPSEEK_API_KEY: str = config("DEEPSEEK_API_KEY", default="")
    DEEPSEEK_BASE_URL: str = config("DEEPSEEK_BASE_URL", default="https://api.deepseek.com")
    QWEN_API_KEY: str = config("QWEN_API_KEY", default="")
    QWEN_BASE_URL: str = config("QWEN_BASE_URL", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    GLM_API_KEY: str = config("GLM_API_KEY", default="")
    GLM_BASE_URL: str = config("GLM_BASE_URL", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ORCHESTRATOR_MODEL: str = config("ORCHESTRATOR_MODEL", default="qwen3.7-max")
    REVIEWER_MODEL: str = config("REVIEWER_MODEL", default="qwen3.7-max")

    DEFAULT_CANDIDATES: int = config("DEFAULT_CANDIDATES", default=3, cast=int)
    MAX_HISTORY_MESSAGES: int = config("MAX_HISTORY_MESSAGES", default=20, cast=int)

    REVIEW_CRITERIA: dict = {
        "logic":         {"label": "逻辑性",   "weight": 0.20},
        "innovation":    {"label": "创新性",   "weight": 0.15},
        "executability": {"label": "可执行性", "weight": 0.20},
        "cost":          {"label": "成本",     "weight": 0.10},
        "risk":          {"label": "风险",     "weight": 0.05},
        "quality":       {"label": "结果质量", "weight": 0.20},
        "explainability":{"label": "可解释性", "weight": 0.10},
    }


settings = Settings()
