from capabilities.base import BaseCapability

PRODUCT_PROMPT = """
在处理这个任务时，请增强产品与用户视角分析能力：

1. 优先关注用户真实需求
2. 分析：
   - 用户痛点
   - 使用场景
   - 用户路径
3. 注意产品功能的：
   - 实用性
   - 易用性
   - 用户体验
4. 避免功能堆砌
5. 优先考虑 MVP（最小可行产品）
6. 分析商业价值与用户价值
7. 如果存在用户体验风险，主动指出
"""


class ProductCapability(BaseCapability):
    name = "产品"
    prompt = PRODUCT_PROMPT
