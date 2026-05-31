"""第二层：意图自适应模式（Intent Classification）

对用户输入进行意图分类，输出 intent + confidence。
规则快判 + LLM 兜底。
"""

import json
import re
from llm_client import call_qianwen
from config import settings

INTENT_CLASSIFY_PROMPT = """你是一个意图分类器。根据用户消息和当前对话上下文，判断用户意图。

意图类型：
1. new_topic — 全新话题，与之前讨论无关
2. followup — 追问、深入、延续之前的分析
3. compare_choice — 对比、选择、权衡多个选项
4. execute — 执行、落地、生成最终内容（写大纲、写代码、生成文档）
5. casual_confirm — 闲聊、确认、轻反馈（嗯、好的、有道理）
6. clarify — 澄清、补充上下文、修正理解

判断原则：
- 如果消息明显延续当前话题，优先 followup
- 如果消息要求"做某事"（写、生成、画），优先 execute
- 如果消息很短且无实质内容，优先 casual_confirm
- 不确定时，优先 followup（保守策略）

输出 JSON：
{"intent": "followup", "confidence": 0.85}
"""

# ── 规则快判 ──

CASUAL_PATTERNS = [
    r'^(嗯|好|行|ok|可以|对|是的|没问[题问]|谢谢|了解|明白|确实|有道理|不错|可以了|就这[样个])\s*[。！!.？?]*$',
    r'^(wow|哈哈|呵呵|嗯嗯|好的好的|了解了解)',
    r'^.{1,3}$',  # 3个字以内的纯短消息
]

COMPARE_PATTERNS = [
    r'(对比|比较|哪个好|优劣|优缺点|选哪个|如何选择|还是|vs|A.*B)',
    r'(权衡|取舍|选.*不选)',
]

EXECUTE_PATTERNS = [
    r'(帮我写|生成|画|做一个|创建|输出.*格式|写个|给我.*代码|实现)',
    r'(大纲|草稿|框架|模板|方案书)',
]

FOLLOWUP_PATTERNS = [
    r'(为什么|怎么|如何|详细|展开|深入|具体|再.*说说|解释一下.*刚才)',
    r'(上面|刚才|之前|你说的|那个)',
]


def _quick_intent_classify(message: str) -> tuple[str, float] | None:
    """规则快判：命中返回 (intent, confidence)，否则 None"""
    text = message.strip()

    for p in CASUAL_PATTERNS:
        if re.search(p, text):
            return ("casual_confirm", 0.90)

    for p in COMPARE_PATTERNS:
        if re.search(p, text):
            return ("compare_choice", 0.80)

    for p in EXECUTE_PATTERNS:
        if re.search(p, text):
            return ("execute", 0.80)

    for p in FOLLOWUP_PATTERNS:
        if re.search(p, text):
            return ("followup", 0.75)

    return None


class IntentClassifier:
    async def classify(
        self, message: str, session_topic: str = "", depth: int = 0
    ) -> dict:
        # 规则快判
        quick = _quick_intent_classify(message)
        if quick:
            return {"intent": quick[0], "confidence": quick[1], "method": "rule"}

        # 第一轮对话默认 new_topic
        if depth == 0:
            return {"intent": "new_topic", "confidence": 0.95, "method": "default"}

        # LLM 分类
        user_msg = f"""当前对话主题: {session_topic or "无"}
对话深度: 第 {depth} 轮

用户消息: {message}

请判断意图，输出 JSON。"""

        try:
            result = await call_qianwen(
                INTENT_CLASSIFY_PROMPT,
                [{"role": "user", "content": user_msg}],
                model=settings.REVIEWER_MODEL,
            )
            data = json.loads(result.strip())
            return {
                "intent": data.get("intent", "followup"),
                "confidence": float(data.get("confidence", 0.5)),
                "method": "llm",
            }
        except (json.JSONDecodeError, Exception):
            return {"intent": "followup", "confidence": 0.5, "method": "fallback"}


intent_classifier = IntentClassifier()
