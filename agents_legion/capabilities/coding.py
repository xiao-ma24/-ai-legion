from capabilities.base import BaseCapability

CODING_PROMPT = """
在处理这个任务时，请增强工程化与代码实现能力：

1. 优先考虑实际可实现性，而不是纯理论方案
2. 给出清晰的模块划分和系统结构
3. 尽量使用：
   - 低耦合
   - 高内聚
   - 易扩展
   - 易维护
   的设计思路
4. 说明关键数据流和模块调用关系
5. 优先考虑：
   - 性能
   - 稳定性
   - 可测试性
   - 异常处理
6. 如果涉及代码：
   - 给出核心逻辑
   - 避免无意义样板代码
7. 如果存在工程风险，主动指出
"""


class CodingCapability(BaseCapability):
    name = "编码"
    prompt = CODING_PROMPT  
