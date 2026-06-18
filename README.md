# 🏛️ AI Legion · 多智能体协同推理系统

> An adaptive multi-agent reasoning system — it routes each question to just enough intelligence.
>
> 一个会"按需调度"的多智能体系统:简单问题一个智能体秒回,复杂问题三个智能体协同 + 评审团打分 + 整合成稿。不为简单问题付复杂的代价。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-async-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/LLM-Qwen%20%7C%20DeepSeek%20%7C%20GLM-orange.svg" alt="LLM">
  <img src="https://img.shields.io/badge/deploy-Railway-purple.svg" alt="Railway">
</p>

---

## ✨ 这是什么

大多数"多智能体"项目的做法是:把同一个问题丢给 N 个模型,然后让它们投票。**简单粗暴,而且贵。**

AI Legion 想解决的核心问题是:**不同的问题,值得不同的"思考成本"。**

- 你问"翻译这句话" → 系统识别为简单任务,**1 个智能体直接回**,不浪费一分钱算力。
- 你问"帮我规划大二的学习路线" → 系统识别为战略级任务,**3 个角色智能体(严谨/发散/批判)并行思考 → 评审团 7 维度打分选最优 → 整合器融合成一份连贯终稿**。

中间还有一整套**意图识别、复杂度分级、动态路由、一致性短路、成本追踪**的机制,在"答得好"和"花得省"之间做权衡。

---

## 🧠 系统架构

```
                              用户消息
                                 │
                                 ▼
┌──────────────────────── L1 · 总控层 ─────────────────────────┐
│  意图分类  (规则快判 + LLM 兜底)  → 6 类意图 + 置信度          │
│      │  置信度 < 0.75 → 保守回退,避免走错链路                │
│      ▼                                                       │
│  路由策略  (意图 × 置信度 × 能力  →  选择哪些智能体)          │
│      │                                                       │
│  复杂度分析 (新话题): simple / analytical / strategic        │
│  预搜索:  🔍 Web 搜索  +  📄 ArXiv 论文  →  注入为上下文      │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────── L2 · 智能体层 ───────────────────────┐
│    🔬 严谨型        🌊 发散型        ⚔️ 批判型   (asyncio 并行) │
│    + 能力插件:  research / writing / coding / product / math  │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────── L3 · 评审整合层 ─────────────────────┐
│  一致性检查  →  双方高度一致则"短路",省下一次评审            │
│  评审团 Judge:  7 维度加权打分 → 选最优 + 提取他者亮点        │
│  整合器 Integrator:  以最优为骨架,融合亮点 → 一份连贯终稿     │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
                  最终回答  +  本次成本(tokens / ¥)
```

---

## 🎯 自适应路由:7 条执行路径

系统不是"一招走天下",而是根据意图与复杂度,走不同深度的链路:

| 意图 / 复杂度 | 执行路径 | 参与智能体 | 成本 |
|---|---|---|---|
| `casual_confirm` / `clarify` | 轻量快回 | 单次 LLM,不进 Pipeline | 💰 |
| `new_topic` · **simple** | 单智能体 | 🔬 严谨型 | 💰 |
| `new_topic` · **analytical** | 双智能体 + 一致性检查 | 🌊 发散 + 🔬 严谨 | 💰💰 |
| `new_topic` · **strategic** | 三智能体 + 评审 + 整合 | 🌊 + 🔬 + ⚔️ | 💰💰💰 |
| `followup` 追问 | 单/双(含风险词时 +批判) + 搜索 | 🔬 (+⚔️) | 💰💰 |
| `compare_choice` 对比 | 双智能体 + 评审 | 🌊 + 🔬 | 💰💰 |
| `execute` 执行落地 | 单智能体 | 🔬 严谨型 | 💰 |

---

## ⚖️ 评审团:7 维度加权打分

复杂任务的多个方案,不靠"谁长谁赢",而是由独立评审团按 7 个维度加权打分,并**额外提取落选方案中值得融合的亮点**交给整合器:

| 维度 | 权重 | 维度 | 权重 |
|---|---|---|---|
| 逻辑性 | 20% | 创新性 | 15% |
| 可执行性 | 20% | 成本 | 10% |
| 结果质量 | 20% | 可解释性 | 10% |
| 风险 | 5% | | |

---

## 💡 设计亮点(为什么这么设计)

这套系统真正花心思的地方,在于**在"答得好"和"花得省"之间做权衡**:

1. **两级分类,能省则省** — 意图分类与复杂度分析都先走正则快判,命中就不调用 LLM。简单请求(翻译/确认/计算)零额外推理成本。
2. **算力按需分配** — `simple` 只用 1 个智能体,`strategic` 才上"三智能体 + 评审 + 整合"。不为简单问题付复杂代价。
3. **一致性短路** — 双智能体结论高度一致(置信度 ≥ 85%)时,直接采用,**跳过评审团**,省一次大模型调用。
4. **不确定性兜底** — 意图置信度 < 0.75 时自动退回保守路径(追问深入),避免误判导致走错链路。
5. **风险感知升级** — 检测到"风险/安全/漏洞/bug"等关键词,自动追加批判型智能体把关。
6. **多模型异构** — 总控与评审用 Qwen,智能体推理用 DeepSeek,可在 `config.py` 中切换(GLM 亦支持)。
7. **成本全程可见** — 每次对话实时累计调用次数、token 数与人民币费用,显示在结果卡片上。

---

## 🛠️ 技术栈

| 层 | 技术 |
|---|---|
| Web 框架 | FastAPI + Uvicorn(全异步) |
| 前端 | Jinja2 + HTMX(轮询式实时进度,无重前端框架) |
| LLM 接入 | OpenAI 兼容客户端 → Qwen(DashScope) / DeepSeek / GLM |
| 编排 | 自研三层架构(L1 总控 / L2 智能体 / L3 评审整合) |
| 工具 | Web 搜索 + ArXiv 论文检索(按需预搜索注入) |
| 记忆 | JSON 文件会话存储 + 会话摘要(主题 / 深度追踪) |
| 可观测 | 内置 token / 费用追踪 |
| 部署 | Railway |

---

## 🚀 快速开始

```bash
# 1. 安装依赖
cd agents_legion
pip install -r requirements.txt

# 2. 配置 API Key —— 在 agents_legion/ 下新建 .env 文件
#    （绝不要把 Key 写进代码或提交到 Git）
cat > .env <<'EOF'
QWEN_API_KEY=sk-你的key
DEEPSEEK_API_KEY=sk-你的key
# 可选:
# GLM_API_KEY=...
# ORCHESTRATOR_MODEL=qwen3.7-max
# REVIEWER_MODEL=qwen3.7-max
EOF

# 3. 启动
python main.py

# 4. 打开浏览器
#    http://127.0.0.1:8000
```

---

## 📁 项目结构

```
agents_legion/
├── main.py               # FastAPI 入口 + 7 条执行路径的编排
├── intent_classifier.py  # L1 意图分类(规则快判 + LLM 兜底)
├── routing_policy.py     # L1 路由策略(意图 × 置信度 × 能力 → 智能体)
├── orchestrator.py       # L1 总控:复杂度分析 / 预搜索 / 并行调度 / 一致性检查
├── agents/               # L2 三种角色智能体
│   ├── base.py           #   统一基类(角色 + 能力插件 + 上下文组装)
│   ├── rigorous_agent.py #   🔬 严谨型(逻辑 / 可行性)
│   ├── divergent_agent.py#   🌊 发散型(创意 / 广度)
│   └── critical_agent.py #   ⚔️ 批判型(风险 / 质疑)
├── capabilities/         # 能力插件:research / writing / coding / product / math
├── reviewer.py           # L3 评审团 Judge(7 维打分)+ 整合器 Integrator
├── memory.py             # 会话记忆(JSON 持久化)
├── summary_manager.py    # 会话摘要(主题 / 对话深度)
├── cost_tracker.py       # token / 费用追踪
├── tools/                # web_search / arxiv_search
├── llm_client.py         # Qwen / DeepSeek / GLM 统一调用(OpenAI 兼容)
├── templates/            # Jinja2 + HTMX 前端
└── static/
```

---

## 🛣️ Roadmap

- [ ] **可训练的路由器** — 当前意图/复杂度路由依赖"正则 + LLM",计划训练一个轻量分类模型替换它,降低延迟与成本(用现有 LLM 路由器做知识蒸馏生成训练数据)。
- [ ] **路由 Benchmark** — 系统对比"LLM 路由 vs 训练模型路由"的准确率、延迟、成本。
- [ ] 评审维度权重可配置化,引入人工标注做校准。
- [ ] 会话摘要的长期记忆压缩。

---

## 📌 项目说明

个人项目,对"多智能体协同推理 + 自适应成本控制"的一次完整探索与工程实现。架构设计、复杂度分级、路由策略、评审维度均为自主设计。

欢迎 Issue 与建议。
