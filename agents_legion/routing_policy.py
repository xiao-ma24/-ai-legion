"""第二层：路由策略（Routing Policy）

根据 intent + complexity + capabilities 决定调用哪些 Agent。
"""

import re

RISK_KEYWORDS = [
    "风险", "安全", "稳定", "可靠", "问题", "漏洞",
    "隐患", "失败", "崩溃", "bug", "error",
]


class RoutingPolicy:
    def get_roles(
        self, intent: str, confidence: float, capabilities: list[str], message: str = ""
    ) -> tuple[list[str], str]:
        """返回 (agent角色列表, 路由策略名)"""
        if confidence < 0.75:
            return self._conservative(capabilities)

        handler = {
            "new_topic": self._new_topic,
            "followup": self._followup,
            "compare_choice": self._compare_choice,
            "execute": self._execute,
            "casual_confirm": self._casual,
            "clarify": self._clarify,
        }.get(intent, self._conservative)

        return handler(capabilities, message)

    def _new_topic(self, capabilities, message=""):
        # new_topic 走完整复杂度分析，由 orchestrator 决定
        return (["full_pipeline"], "new_topic_complexity")

    def _followup(self, capabilities, message=""):
        has_risk = any(kw in message for kw in RISK_KEYWORDS)
        if has_risk:
            return (["rigorous", "critical"], "followup_with_risk")
        return (["rigorous"], "followup")

    def _compare_choice(self, capabilities, message=""):
        return (["divergent", "rigorous"], "compare")

    def _execute(self, capabilities, message=""):
        if "coding" in capabilities:
            return (["rigorous"], "execute_coding")
        return (["rigorous"], "execute")

    def _casual(self, capabilities, message=""):
        return ([], "lightweight")

    def _clarify(self, capabilities, message=""):
        return ([], "lightweight")

    def _conservative(self, capabilities, message=""):
        return (["divergent", "rigorous"], "conservative_analytical")


routing_policy = RoutingPolicy()
