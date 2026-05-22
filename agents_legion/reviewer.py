import json
from config import settings
from llm_client import call_qianwen

JUDGE_PROMPT_BASE = """你是一个专业级 AI 评审专家。

你的职责不是简单打分，
而是系统性评估多个候选方案的：

- 推理质量
- 实际价值
- 可执行性
- 风险控制
- 创新程度
- 整体完成度

然后选出当前任务下最优的方案。

====================================
评审原则
====================================

1. 不以“内容长度”作为质量标准
2. 不偏向华丽表达，而优先关注：
   - 逻辑
   - 可执行性
   - 推理完整性
3. 不允许机械平均打分
4. 不同方案可以在不同维度上各有优势
5. 如果某方案存在明显逻辑漏洞，必须明确指出
6. 必须真正做出判断，而不是模糊中立
7. 允许：
   - 某个方案创新极强但风险较高
   - 某个方案稳健但缺乏突破
8. 评分必须有明确理由

评分维度：
"""

INTEGRATOR_PROMPT =  """
你是一个高级内容整合专家。

你的任务是基于多个候选方案，
生成一个逻辑统一、结构完整、表达自然的最终答案。

整合原则：
1. 以「最优方案」作为主体框架
2. 仅吸收其他方案中真正有价值的补充观点
3. 避免机械拼接和重复表达
4. 保持整体逻辑一致，避免前后冲突
5. 删除冗余、空洞或低价值内容
6. 最终输出必须像“一个人完整思考后的结果”，而不是多个答案的堆叠

工作方式：
1. 先识别最优方案的核心结构
2. 提取其他方案中的：
   - 独到视角
   - 更强论据
   - 更完整细节
   - 更高质量表达
3. 判断这些内容是否真的增强最终答案
4. 如果补充内容会破坏逻辑一致性，则舍弃
5. 对整合后的内容重新组织语言与结构

输出要求：
- 保持自然流畅
- 不出现“方案A认为”“另一个方案提到”等痕迹
- 不解释整合过程
- 不输出元信息
- 直接输出最终高质量答案

你的目标：
生成一个：
- 更全面
- 更严谨
- 更自然
- 更高质量

且明显优于任何单一候选方案的最终答案。
"""

class Judge:
    """L3 裁判：多维度评分，选出最优方案"""

    def __init__(self):
        self.criteria = settings.REVIEW_CRITERIA

    def build_prompt(self) -> str:
        lines = [JUDGE_PROMPT_BASE]
        for key, info in self.criteria.items():
            lines.append(
                f"- **{info['label']}**（权重 {info['weight']*100:.0f}%）"
            )
        lines.append("""
请对每个候选方案逐维度打分（1-10分），计算加权总分，选出最优方案,提取其他方案中值得融合的亮点。
--将每一个智能体的输出内容视为一个独立的候选方案，进行全面评审，而不是简单比较哪个更“好”。
--评审时要真正理解每个方案的核心逻辑和细节，而不是停留在表面印象
--将每个智能体方案的优点提取出来，输出为“highlights_from_others”，以便后续整合器参考，而不是简单地选出一个赢家就结束评审。

输出 JSON 格式：
{
  "scores": [
    {"agent": "发散型", "scores": {"logic": 8, "innovation": 9}, "total": 8.2},
    {"agent": "严谨型", "scores": {"logic": 9, "innovation": 6}, "total": 7.8},
    {"agent": "批判型", "scores": {"logic": 7, "innovation": 7}, "total": 7.5}
  ],
  "winner": "严谨型",
  "winner_reason": "逻辑最严密，可执行性最强",
  "highlights_from_others": ["发散型的新颖角度", "批判型发现的风险点"]
}
""")
        return "\n".join(lines)

    async def evaluate(self, outputs: list[dict]) -> dict:
        prompt = self.build_prompt()
        content = "\n\n---\n\n".join([
            f"### 候选方案 {i+1}（{o['agent']}）\n{o['content']}"
            for i, o in enumerate(outputs)
        ])
        result = await call_qianwen(
            prompt,
            [{"role": "user", "content": content}],
            model=settings.REVIEWER_MODEL,
        )
        try:
            return json.loads(result.strip())
        except json.JSONDecodeError:
            return {
                "winner": outputs[0]["agent"],
                "highlights_from_others": [],
                "scores": [],
            }


class Integrator:
    """L3 整合器：以最优方案为主体，融入其他方案亮点"""

    async def integrate(
        self,
        task: str,
        winner_name: str,
        winner_content: str,
        highlights: list[str],
    ) -> str:
        highlights_text = "\n".join([f"- {h}" for h in highlights])
        user_msg = f"""## 用户原始任务
{task}

## 最优方案（来自 {winner_name}）——以此为主体
{winner_content}

## 其他方案的亮点——选择性融入
{highlights_text}
"""
        return await call_qianwen(
            INTEGRATOR_PROMPT,
            [{"role": "user", "content": user_msg}],
            model=settings.REVIEWER_MODEL,
        )
