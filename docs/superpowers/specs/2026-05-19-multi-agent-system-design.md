# 多智能体协同系统 — 设计文档

## 概述

构建一个基于"三层+按需派生"架构的多智能体协同工作系统，通过 Web 聊天界面交互，帮助大学生处理学习、竞赛、论文等需要强推理的任务。

## 技术栈

| 层 | 技术 | 理由 |
|---|---|---|
| 后端 | FastAPI (Python) | 异步、流式、AI 生态原生，用户会 Python |
| 模板 | Jinja2 | 服务端渲染，用户看得懂 |
| 交互 | HTMX + SSE | 声明式，无需 JS |
| 样式 | Tailwind CSS (CDN) | 响应式，移动端友好 |
| LLM | Claude API + DeepSeek API | 云端按量付费，学生友好 |
| 存储 | SQLite + JSON | 零配置便携 |
| 部署 | 单进程 Python | `python main.py` 即跑 |

## 三层架构

### L1 总控层 (orchestrator.py)
- 判断任务类型（单任务 / 多任务）
- 决定候选方案数量
- 匹配角色 × 能力模板组合
- 控制预算（是否走多模型高成本流程）

### L2 任务执行层 (agents/)
- **3 个通用角色**：发散型 (divergent) / 严谨型 (rigorous) / 批判型 (critical)
- **4 个能力模板**：写作 (writing) / 代码 (coding) / 研究 (research) / 视觉 (vision)
- 角色 × 能力 = 按需组合，按 L1 决策并行执行

### L3 评审整合层 (reviewer.py)
- **裁判**：7 维度加权评分（逻辑性、创新性、可执行性、成本、风险、结果质量、可解释性）
- **整合器**：融合最优内容，输出最终答案

### 经验库 (memory.py)
- 记录成功路径（任务类型 → 模板组合 → 评分）
- 下次类似任务直接复用

## 工程结构

```
agents_legion/
├── main.py              # FastAPI 入口 + SSE 路由
├── config.py            # API Key、模型配置
├── orchestrator.py      # L1 总控
├── agents/              # L2 角色定义
│   ├── __init__.py
│   ├── base.py          # 角色基类
│   ├── divergent_agent.py
│   ├── rigorous_agent.py
│   └── critical_agent.py
├── capabilities/        # L2 能力模板
│   ├── __init__.py
│   ├── base.py
│   ├── writing.py
│   ├── coding.py
│   ├── research.py
│   └── vision.py
├── reviewer.py          # L3 裁判 + 整合器
├── memory.py            # 经验库
├── templates/           # Jinja2 HTML
│   ├── base.html
│   ├── chat.html
│   └── components/
├── static/
├── data/
└── requirements.txt
```

## 核心流程

1. 用户通过 Web 聊天界面输入任务
2. L1 总控分析任务 → 判断单/多任务 → 选择角色×能力组合 → 决定预算
3. L2 执行层并行调用 Claude/DeepSeek API（不同角色使用不同 system prompt）
4. L3 评审层对多输出打分 → 整合器融合 → 输出最终答案
5. 经验库记录成功路径
6. SSE 流式推送进度，前端逐段渲染

## 一期范围

- 核心 Web 聊天界面（响应式，移动端可用）
- 三层架构完整链路（L1 → L2 → L3 → 输出）
- 发散型 + 严谨型 + 批判型三个角色
- 研究 + 写作两个能力模板（先覆盖论文/竞赛场景）
- 单任务三路并行评审流程
- 后端 SSE 流式推送
- 会话内对话记忆

## 后续迭代

- 多任务拆解分发流程
- 视觉/代码能力模板
- 文件上传 + 图片解析
- 工具调用（ArXiv、联网搜索、代码执行）
- 经验库自动匹配复用
