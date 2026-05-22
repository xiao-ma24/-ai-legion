from capabilities.base import BaseCapability

WRITING_PROMPT = """
在处理这个任务时，请增强专业写作能力：

1. 保持表达清晰、自然、结构化
2. 优先确保逻辑流畅与阅读体验
3. 注意段落衔接与上下文一致性
4. 避免重复表达和空洞套话
5. 根据内容类型调整语言风格：
   - 学术
   - 专业
   - 通俗
   - 商业
6. 重要观点尽量简洁有力
7. 输出应具有较强可读性与完整性
"""


class WritingCapability(BaseCapability):
    name = "写作"
    prompt = WRITING_PROMPT
