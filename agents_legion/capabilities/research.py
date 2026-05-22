from capabilities.base import BaseCapability

RESEARCH_PROMPT = """
在处理这个任务时，请增强研究型分析能力：

1. 优先基于事实、数据和可验证信息进行分析
2. 明确区分：
   - 已知事实
   - 推测
   - 假设
3. 如果某个信息不确定，明确标注“未验证”
4. 尽量说明：
   - 信息依据
   - 推理来源
   - 结论成立条件
5. 优先引用：
   - 官方资料
   - 学术研究
   - 权威来源
6. 注意知识边界，避免过度确定性表达
7. 在关键结论处，简单标注可信度或不确定性
"""


class ResearchCapability(BaseCapability):
    name = "研究"
    prompt = RESEARCH_PROMPT
