"""成本追踪：记录每次 LLM 调用的 token 消耗和费用"""

import time
from collections import defaultdict

# 模型价格表（¥/千tokens）
# 数据来源：各平台官方定价页，请定期核对更新
PRICES = {
    # 阿里云百炼 DashScope
    "qwen3.7-max":     {"input": 0.008, "output": 0.024},
    "qwen-plus":       {"input": 0.002, "output": 0.006},
    "qwen-turbo":      {"input": 0.0003, "output": 0.0006},
    # DeepSeek 官方 API
    "deepseek-v4-pro": {"input": 0.002, "output": 0.008},
    "deepseek-chat":   {"input": 0.001, "output": 0.002},
    # 智谱 GLM（按 DashScope 通道价格估算）
    "glm-5.1":         {"input": 0.008, "output": 0.024},
    "glm-4-plus":      {"input": 0.050, "output": 0.050},
}
FALLBACK_PRICE = {"input": 0.010, "output": 0.030}  # 未知模型默认价


class CostTracker:
    def __init__(self):
        self._tasks: dict[str, dict] = {}

    def record(self, task_id: str, model: str, input_tokens: int, output_tokens: int, duration_s: float):
        if task_id not in self._tasks:
            self._tasks[task_id] = {"calls": [], "total_cost": 0.0, "total_tokens": 0}
        price = PRICES.get(model, FALLBACK_PRICE)
        cost = (
            price["input"] * input_tokens / 1000 +
            price["output"] * output_tokens / 1000
        )
        self._tasks[task_id]["calls"].append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration": duration_s,
            "cost": cost,
        })
        self._tasks[task_id]["total_cost"] += cost
        self._tasks[task_id]["total_tokens"] += input_tokens + output_tokens

    def get_summary(self, task_id: str) -> dict | None:
        t = self._tasks.get(task_id)
        if not t:
            return None
        return {
            "calls": len(t["calls"]),
            "total_cost": t["total_cost"],
            "total_tokens": t["total_tokens"],
            "models": list(set(c["model"] for c in t["calls"])),
        }

    def cleanup(self, task_id: str):
        self._tasks.pop(task_id, None)


cost_tracker = CostTracker()
