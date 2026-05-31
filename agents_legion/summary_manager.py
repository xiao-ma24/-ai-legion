"""第一层：对话上下文感知（Session Summary）

管理结构化的会话摘要，让系统知道：
- 当前在讨论什么
- 已经分析到哪里
- 哪些内容已经确认
- 哪些问题还没解决
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from llm_client import call_qianwen
from config import settings

SUMMARY_UPDATE_PROMPT = """你是一个对话状态追踪器。

根据当前摘要和新一轮对话，更新摘要。

要求：
1. 保持简洁，每个字段尽量精炼
2. key_decisions 只保留最重要的 5 条
3. confirmed_choices 只保留已明确确认的
4. open_questions 只保留当前仍未解决的
5. current_stage 必须是以下之一：
   - 发散阶段 / 收敛阶段 / 对比阶段 / 执行阶段 / 完成阶段 / 初始阶段
6. depth 每次 +1
7. 如果话题明显变化，重置相关字段

输出 JSON：
{
  "current_topic": "当前讨论主题",
  "current_stage": "当前阶段",
  "key_decisions": ["关键决策1", "关键决策2"],
  "confirmed_choices": ["已确认的选择"],
  "open_questions": ["待解决的问题"],
  "last_winner": "上一轮最优方案的agent名",
  "last_intent": "上一轮的意图类型",
  "depth": 2
}
"""


@dataclass
class SessionSummary:
    current_topic: str = ""
    current_stage: str = "初始阶段"
    key_decisions: list[str] = field(default_factory=list)
    confirmed_choices: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    last_winner: str = ""
    last_intent: str = ""
    depth: int = 0

    def to_context_string(self) -> str:
        if self.depth == 0:
            return ""
        parts = [
            f"当前主题: {self.current_topic}",
            f"当前阶段: {self.current_stage}",
            f"讨论深度: 第 {self.depth} 轮",
        ]
        if self.key_decisions:
            parts.append(f"关键决策: {'; '.join(self.key_decisions[:3])}")
        if self.confirmed_choices:
            parts.append(f"已确认: {'; '.join(self.confirmed_choices)}")
        if self.open_questions:
            parts.append(f"待解决: {'; '.join(self.open_questions)}")
        if self.last_winner:
            parts.append(f"上轮最优方案来源: {self.last_winner}")
        if self.last_intent:
            parts.append(f"上轮意图: {self.last_intent}")
        return "\n".join(parts)


class SummaryManager:
    def __init__(self):
        self._summaries: dict[str, SessionSummary] = {}

    def get(self, session_id: str) -> SessionSummary:
        if session_id not in self._summaries:
            self._summaries[session_id] = SessionSummary()
        return self._summaries[session_id]

    async def update(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        intent: str,
        winner: str = "",
    ) -> SessionSummary:
        old_summary = self.get(session_id)
        old_json = json.dumps(asdict(old_summary), ensure_ascii=False, indent=2)

        user_msg = f"""## 当前摘要
{old_json}

## 用户新消息
{user_message}

## 系统回复摘要（前500字）
{assistant_response[:500]}

## 本轮意图
{intent}

## 本轮最优方案来源
{winner or "无"}

请输出更新后的摘要 JSON。"""

        try:
            result = await call_qianwen(
                SUMMARY_UPDATE_PROMPT,
                [{"role": "user", "content": user_msg}],
                model=settings.REVIEWER_MODEL,
            )
            data = json.loads(result.strip())
            new_summary = SessionSummary(
                current_topic=data.get("current_topic", old_summary.current_topic),
                current_stage=data.get("current_stage", old_summary.current_stage),
                key_decisions=data.get("key_decisions", old_summary.key_decisions)[:5],
                confirmed_choices=data.get("confirmed_choices", old_summary.confirmed_choices),
                open_questions=data.get("open_questions", old_summary.open_questions),
                last_winner=winner or old_summary.last_winner,
                last_intent=intent,
                depth=data.get("depth", old_summary.depth + 1),
            )
        except (json.JSONDecodeError, Exception):
            new_summary = SessionSummary(
                current_topic=old_summary.current_topic or user_message[:30],
                current_stage=old_summary.current_stage,
                key_decisions=old_summary.key_decisions,
                confirmed_choices=old_summary.confirmed_choices,
                open_questions=old_summary.open_questions,
                last_winner=winner or old_summary.last_winner,
                last_intent=intent,
                depth=old_summary.depth + 1,
            )

        self._summaries[session_id] = new_summary
        return new_summary

    def reset(self, session_id: str):
        self._summaries[session_id] = SessionSummary()


summary_manager = SummaryManager()
