from capabilities.base import BaseCapability

MATH_PROMPT = """
在处理这个任务时，请增强数学与形式化推理能力：

1. 优先使用严格逻辑推导
2. 明确：
   - 条件
   - 假设
   - 推导过程
3. 避免跳步结论
4. 如果存在多个解法，分析其优劣
5. 注意边界条件与特殊情况
6. 尽量将问题形式化表达
7. 如果结论依赖某些前提，明确指出
"""


class MathCapability(BaseCapability):
    name = "数学"
    prompt = MATH_PROMPT
