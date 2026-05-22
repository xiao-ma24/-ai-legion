# 多智能体协同系统 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基于"三层+按需派生"架构的多智能体协同 Web 应用，一期实现单任务三路并行评审流程

**Architecture:** FastAPI 后端接收用户任务 → L1 总控分析分类 → L2 三个角色 Agent 并行调用 LLM → L3 裁判评分+整合器融合 → SSE 流式返回前端。会话管理用内存模式，一期无数据库。

**Tech Stack:** Python 3.10+ / FastAPI / Jinja2 / HTMX 2.0 / Tailwind CSS CDN / Anthropic SDK / OpenAI SDK / sse-starlette / uvicorn

---

### Task 1: 项目脚手架

**Files:**
- Create: `agents_legion/requirements.txt`
- Create: `agents_legion/config.py`
- Create: `agents_legion/main.py` (最小入口)
- Create: `agents_legion/agents/__init__.py`
- Create: `agents_legion/capabilities/__init__.py`
- Create: `agents_legion/templates/base.html`
- Create: `agents_legion/templates/chat.html`
- Create: `agents_legion/static/.gitkeep`
- Create: `agents_legion/data/.gitkeep`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p agents_legion/agents agents_legion/capabilities agents_legion/templates/components agents_legion/static agents_legion/data
```

- [ ] **Step 2: 编写 requirements.txt**

```txt
fastapi==0.115.6
uvicorn[standard]==0.34.0
jinja2==3.1.4
python-multipart==0.0.18
sse-starlette==2.2.1
anthropic==0.42.0
openai==1.58.1
python-decouple==3.8
```

- [ ] **Step 3: 编写 config.py**

```python
import os
from decouple import config, Csv


class Settings:
    ANTHROPIC_API_KEY: str = config("ANTHROPIC_API_KEY", default="")
    DEEPSEEK_API_KEY: str = config("DEEPSEEK_API_KEY", default="")
    DEEPSEEK_BASE_URL: str = config("DEEPSEEK_BASE_URL", default="https://api.deepseek.com")

    ORCHESTRATOR_MODEL: str = config("ORCHESTRATOR_MODEL", default="claude-sonnet-4-6")
    AGENT_MODEL: str = config("AGENT_MODEL", default="claude-sonnet-4-6")
    REVIEWER_MODEL: str = config("REVIEWER_MODEL", default="claude-sonnet-4-6")

    DEFAULT_CANDIDATES: int = config("DEFAULT_CANDIDATES", default=3, cast=int)
    MAX_HISTORY_MESSAGES: int = config("MAX_HISTORY_MESSAGES", default=20, cast=int)

    REVIEW_CRITERIA: dict = {
        "logic":         {"label": "逻辑性",   "weight": 0.20},
        "innovation":    {"label": "创新性",   "weight": 0.15},
        "executability": {"label": "可执行性", "weight": 0.15},
        "cost":          {"label": "成本",     "weight": 0.10},
        "risk":          {"label": "风险",     "weight": 0.10},
        "quality":       {"label": "结果质量", "weight": 0.20},
        "explainability":{"label": "可解释性", "weight": 0.10},
    }


settings = Settings()
```

- [ ] **Step 4: 编写最小 main.py**

```python
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AI 军团")

@app.get("/")
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/chat")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
```

- [ ] **Step 5: 验证脚手架可运行**

```bash
cd agents_legion && python main.py
```
Expected: FastAPI 启动在 http://127.0.0.1:8000，访问 `/` 重定向到 `/chat`（后者暂 404）。

- [ ] **Step 6: 提交**

```bash
git add agents_legion/
git commit -m "feat: project scaffold with config and minimal FastAPI entry"
```

---

### Task 2: 会话记忆模块

**Files:**
- Create: `agents_legion/memory.py`

- [ ] **Step 1: 编写 memory.py**

```python
from datetime import datetime
from collections import OrderedDict


class SessionMemory:
    def __init__(self, max_messages: int = 20):
        self._sessions: dict[str, list[dict]] = {}
        self.max_messages = max_messages

    def add(self, session_id: str, role: str, content: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({
            "role": role,
            "content": content,
            "time": datetime.now().isoformat(),
        })
        if len(self._sessions[session_id]) > self.max_messages:
            self._sessions[session_id] = self._sessions[session_id][-self.max_messages:]

    def get_history(self, session_id: str, last_n: int | None = None) -> list[dict]:
        msgs = self._sessions.get(session_id, [])
        if last_n:
            return msgs[-last_n:]
        return msgs

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)


memory = SessionMemory()
```

- [ ] **Step 2: 交互验证**

```bash
cd agents_legion && python -c "
from memory import memory
memory.add('test', 'user', 'hello')
memory.add('test', 'assistant', 'hi there')
print(memory.get_history('test'))
print('OK')
"
```
Expected: 输出两行消息 JSON 和 OK

---

### Task 3: LLM 调用抽象层 + Agent 基类

**Files:**
- Create: `agents_legion/llm_client.py`
- Create: `agents_legion/agents/base.py`
- Create: `agents_legion/capabilities/base.py`

- [ ] **Step 1: 编写 llm_client.py**

```python
import anthropic
from openai import AsyncOpenAI
from config import settings


async def call_claude(system_prompt: str, messages: list[dict], model: str | None = None) -> str:
    model = model or settings.AGENT_MODEL
    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
    )
    return response.content[0].text


async def call_deepseek(system_prompt: str, messages: list[dict], model: str = "deepseek-chat") -> str:
    client = AsyncOpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
    )
    formatted = [{"role": "system", "content": system_prompt}]
    for m in messages:
        formatted.append({"role": m["role"], "content": m["content"]})

    response = await client.chat.completions.create(
        model=model,
        messages=formatted,
        max_tokens=4096,
    )
    return response.choices[0].message.content
```

- [ ] **Step 2: 编写 agents/base.py**

```python
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, name: str, role_prompt: str):
        self.name = name
        self.role_prompt = role_prompt
        self._capability_prompts: list[str] = []

    def attach_capability(self, prompt: str):
        self._capability_prompts.append(prompt)

    def build_system_prompt(self) -> str:
        parts = [self.role_prompt]
        if self._capability_prompts:
            parts.append("\n## 能力增强\n")
            for i, cp in enumerate(self._capability_prompts, 1):
                parts.append(f"### 能力 {i}\n{cp}")
        return "\n\n".join(parts)

    @abstractmethod
    async def run(self, task: str, context: list[dict], model: str | None = None) -> dict:
        ...
```

- [ ] **Step 3: 编写 capabilities/base.py**

```python
class BaseCapability:
    name: str = ""
    prompt: str = ""

    def get_prompt(self) -> str:
        return self.prompt
```

- [ ] **Step 4: 验证基类可导入**

```bash
cd agents_legion && python -c "
from agents.base import BaseAgent
from capabilities.base import BaseCapability
print('OK')
"
```

---

### Task 4: 三个角色 Agent

**Files:**
- Create: `agents_legion/agents/divergent_agent.py`
- Create: `agents_legion/agents/rigorous_agent.py`
- Create: `agents_legion/agents/critical_agent.py`

- [ ] **Step 1: 编写 divergent_agent.py**

```python
from agents.base import BaseAgent
from llm_client import call_claude

DIVERGENT_PROMPT = """你是一个发散型思维专家。你的核心能力是广度探索和多角度联想。

工作方式：
1. 从多个不同维度审视问题，不局限于单一框架
2. 提出 3-5 个截然不同的方向或解决方案
3. 每个方向都要有独特的切入点和新颖的视角
4. 鼓励跨学科联想，把其他领域的思路迁移过来
5. 不过早收敛，保持思维开放

输出格式：
- 每个方向用 ## 标题分隔
- 每个方向下包含：核心思路、关键假设、潜在优势、可能挑战
"""


class DivergentAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="发散型", role_prompt=DIVERGENT_PROMPT)

    async def run(self, task: str, context: list[dict], model: str | None = None) -> dict:
        system_prompt = self.build_system_prompt()
        messages = [*context, {"role": "user", "content": task}]
        result = await call_claude(system_prompt, messages, model)
        return {"agent": self.name, "content": result}
```

- [ ] **Step 2: 编写 rigorous_agent.py**

```python
from agents.base import BaseAgent
from llm_client import call_deepseek

RIGOROUS_PROMPT = """你是一个严谨型逻辑推理专家。你的核心能力是严密推导和步骤验证。

工作方式：
1. 每一步推理都必须有明确的逻辑依据
2. 从基本假设出发，逐步推演，不能跳步骤
3. 对每个中间结论进行自检：是否有漏洞？是否有反例？
4. 如果涉及数据，进行保守估计并说明误差范围
5. 最终结论必须能被追溯到最初的假设

输出格式：
- 先列出前提假设
- 然后按步骤编号推导
- 每一步标注该步依赖的前提
- 最后总结确定性程度（确定 / 很可能 / 待验证）
"""


class RigorousAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="严谨型", role_prompt=RIGOROUS_PROMPT)

    async def run(self, task: str, context: list[dict], model: str | None = None) -> dict:
        system_prompt = self.build_system_prompt()
        messages = [*context, {"role": "user", "content": task}]
        result = await call_deepseek(system_prompt, messages, model or "deepseek-chat")
        return {"agent": self.name, "content": result}
```

- [ ] **Step 3: 编写 critical_agent.py**

```python
from agents.base import BaseAgent
from llm_client import call_claude

CRITICAL_PROMPT = """你是一个批判型思维专家。你的核心能力是找漏洞和挑毛病。

工作方式：
1. 对给定的任何方案、论证、结论进行系统性挑战
2. 找到隐藏的假设和逻辑跳跃
3. 指出最坏情况下的风险
4. 提出对方没有考虑到的反例或边界条件
5. 不为了否定而否定，批判的目的是让方案更牢固

输出格式：
- 先用表格列出发现的问题（类型 | 严重程度 | 说明）
- 然后逐条详细分析
- 最后给出修正建议（如果方案能修好的话）
"""


class CriticalAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="批判型", role_prompt=CRITICAL_PROMPT)

    async def run(self, task: str, context: list[dict], model: str | None = None) -> dict:
        system_prompt = self.build_system_prompt()
        messages = [*context, {"role": "user", "content": task}]
        result = await call_claude(system_prompt, messages, model)
        return {"agent": self.name, "content": result}
```

- [ ] **Step 4: 更新 agents/__init__.py**

```python
from agents.divergent_agent import DivergentAgent
from agents.rigorous_agent import RigorousAgent
from agents.critical_agent import CriticalAgent

ALL_AGENTS = [DivergentAgent, RigorousAgent, CriticalAgent]
```

- [ ] **Step 5: 单元验证**

```bash
cd agents_legion && python -c "
from agents.divergent_agent import DivergentAgent
from agents.rigorous_agent import RigorousAgent
from agents.critical_agent import CriticalAgent

d = DivergentAgent()
r = RigorousAgent()
c = CriticalAgent()
assert d.name == '发散型'
assert r.name == '严谨型'
assert c.name == '批判型'
print('All agents created OK')
"
```

---

### Task 5: 能力模板（研究 + 写作）

**Files:**
- Create: `agents_legion/capabilities/research.py`
- Create: `agents_legion/capabilities/writing.py`

- [ ] **Step 1: 编写 capabilities/research.py**

```python
from capabilities.base import BaseCapability

RESEARCH_PROMPT = """你需要运用专业的研究方法论来处理这个任务：

1. **信息检索策略**：说明你查找了哪类资料、用什么关键词
2. **来源评估**：对引用的信息标注可靠性（一手来源 / 权威二手 / 仅供参考）
3. **证据链完整性**：从数据到结论的每一环都要可追溯
4. **知识边界标注**：如果不确定某个事实，明确标注"未验证"
5. **阶段性总结**：在每个关键结论处用 ⚠️ 标注不确定性等级
"""


class ResearchCapability(BaseCapability):
    name = "研究"
    prompt = RESEARCH_PROMPT
```

- [ ] **Step 2: 编写 capabilities/writing.py**

```python
from capabilities.base import BaseCapability

WRITING_PROMPT = """你需要运用专业的学术写作规范：

1. **结构化表达**：采用"总-分-总"结构，先给结论再展开
2. **段落逻辑链条**：每段首句陈述论点，中间铺证据，末句承上启下
3. **学术用语**：用词精准、客观，避免口语化表达
4. **引用规范**：涉及他人成果时用 [来源] 标注
5. **可读性**：长句拆短，专业术语首次出现时给出解释
"""


class WritingCapability(BaseCapability):
    name = "写作"
    prompt = WRITING_PROMPT
```

- [ ] **Step 3: 更新 capabilities/__init__.py**

```python
from capabilities.research import ResearchCapability
from capabilities.writing import WritingCapability

CAPABILITY_MAP = {
    "research": ResearchCapability,
    "writing": WritingCapability,
}
```

- [ ] **Step 4: 验证能力模板**

```bash
cd agents_legion && python -c "
from capabilities.research import ResearchCapability
from capabilities.writing import WritingCapability
r = ResearchCapability()
w = WritingCapability()
assert r.name == '研究'
assert w.name == '写作'
# Test attach to agent
from agents.divergent_agent import DivergentAgent
d = DivergentAgent()
d.attach_capability(r.get_prompt())
d.attach_capability(w.get_prompt())
sp = d.build_system_prompt()
assert '发散型' in sp
assert '研究' not in sp  # capability prompt doesn't contain the name directly, but its content
print('Capabilities attached OK')
print(f'System prompt length: {len(sp)} chars')
"
```

---

### Task 6: L1 总控层

**Files:**
- Create: `agents_legion/orchestrator.py`

- [ ] **Step 1: 编写 orchestrator.py**

```python
import asyncio
from config import settings
from agents import DivergentAgent, RigorousAgent, CriticalAgent
from capabilities import CAPABILITY_MAP
from llm_client import call_claude

ORCHESTRATOR_PROMPT = """你是一个任务分析器。你只需要输出 JSON，不要输出其他内容。

分析用户的输入，判断：
- task_type: "single" 或 "multi"（本期默认为 single）
- candidates: 建议并行生成的候选方案数量（2-3）
- capabilities: 需要的能力列表 ["research", "writing"]
- budget: "low" / "normal" / "high"

输出格式严格按照：
{"task_type": "single", "candidates": 3, "capabilities": ["research", "writing"], "budget": "normal"}
"""


class Orchestrator:
    def __init__(self):
        self.agent_classes = {
            "divergent": DivergentAgent,
            "rigorous": RigorousAgent,
            "critical": CriticalAgent,
        }

    async def analyze(self, task: str) -> dict:
        import json
        result = await call_claude(
            ORCHESTRATOR_PROMPT,
            [{"role": "user", "content": task}],
            model=settings.ORCHESTRATOR_MODEL
        )
        try:
            plan = json.loads(result.strip())
        except json.JSONDecodeError:
            plan = {"task_type": "single", "candidates": 3, "capabilities": ["research"], "budget": "normal"}
        return plan

    async def dispatch(self, task: str, plan: dict, context: list[dict], event_callback) -> list[dict]:
        roles = ["divergent", "rigorous", "critical"]
        capability_names = plan.get("capabilities", ["research"])

        agents = []
        for role_name in roles:
            agent_cls = self.agent_classes[role_name]
            agent = agent_cls()
            for cap_name in capability_names:
                cap_cls = CAPABILITY_MAP.get(cap_name)
                if cap_cls:
                    agent.attach_capability(cap_cls().get_prompt())
            agents.append(agent)

        await event_callback("thinking", f"已创建 {len(agents)} 个智能体，开始并行执行...")

        async def run_one(agent):
            await event_callback("agent_start", f"**{agent.name}** 正在分析...")
            result = await agent.run(task, context)
            await event_callback("agent_done", f"**{agent.name}** 分析完成")
            return result

        results = await asyncio.gather(*[run_one(a) for a in agents])
        return results
```

- [ ] **Step 2: 验证总控逻辑**

```bash
cd agents_legion && python -c "
from orchestrator import Orchestrator
o = Orchestrator()
assert 'divergent' in o.agent_classes
assert 'rigorous' in o.agent_classes
assert 'critical' in o.agent_classes
print('Orchestrator initialized OK')
"
```

---

### Task 7: L3 评审整合层

**Files:**
- Create: `agents_legion/reviewer.py`

- [ ] **Step 1: 编写 reviewer.py**

```python
from config import settings
from llm_client import call_claude

JUDGE_PROMPT = """你是一个专业评审。你需要对以下候选方案进行多维度评分。

评分维度：
"""
# build_judge_prompt() dynamically constructs the criteria list

INTEGRATOR_PROMPT = """你是一个内容整合专家。

你的任务是：
1. 收到一份"最优方案"和"其他方案的补充亮点"
2. 以最优方案为主体框架
3. 把其他方案的独到见解和亮点融入进去
4. 保持逻辑流畅，不要机械拼接
5. 输出完整的最终答案

输出直接是最终答案内容，不要加前缀说明。
"""


class Judge:
    def __init__(self):
        self.criteria = settings.REVIEW_CRITERIA

    def build_prompt(self) -> str:
        lines = [JUDGE_PROMPT]
        for key, info in self.criteria.items():
            lines.append(f"- **{info['label']}**（权重 {info['weight']*100:.0f}%）：评估{info['label']}水平")
        lines.append("""
请对每个候选方案逐维度打分（1-10分），然后计算加权总分。

输出 JSON 格式：
{
  "scores": [
    {"agent": "发散型", "scores": {"logic": 8, "innovation": 9, ...}, "total": 8.2},
    ...
  ],
  "winner": "严谨型",
  "winner_reason": "逻辑性最强，成本控制最优",
  "highlights_from_others": ["发散型的新颖角度 X", "批判型发现的风险点 Y"]
}
""")
        return "\n".join(lines)

    async def evaluate(self, outputs: list[dict]) -> dict:
        import json
        prompt = self.build_prompt()
        content = "\n\n---\n\n".join([
            f"### 候选方案 {i+1}（{o['agent']}）\n{o['content']}"
            for i, o in enumerate(outputs)
        ])
        result = await call_claude(prompt, [{"role": "user", "content": content}], model=settings.REVIEWER_MODEL)
        try:
            return json.loads(result.strip())
        except json.JSONDecodeError:
            return {"winner": outputs[0]["agent"], "highlights_from_others": [], "scores": []}


class Integrator:
    async def integrate(self, task: str, winner_name: str, winner_content: str, highlights: list[str]) -> str:
        highlights_text = "\n".join([f"- {h}" for h in highlights])
        full_prompt = INTEGRATOR_PROMPT
        user_msg = f"""## 用户原始任务
{task}

## 最优方案（来自{winner_name}）- 请以此为主体
{winner_content}

## 其他方案的亮点 - 请选择性融入
{highlights_text}
"""
        result = await call_claude(full_prompt, [{"role": "user", "content": user_msg}], model=settings.REVIEWER_MODEL)
        return result
```

- [ ] **Step 2: 验证评审模块**

```bash
cd agents_legion && python -c "
from reviewer import Judge, Integrator
j = Judge()
assert len(j.criteria) == 7
print(f'Judge criteria: {list(j.criteria.keys())}')
i = Integrator()
print('Reviewer modules OK')
"
```

---

### Task 8: FastAPI 主应用 + SSE 流式端点

**Files:**
- Modify: `agents_legion/main.py` (重写)
- Create: `agents_legion/sse_manager.py`

- [ ] **Step 1: 编写 sse_manager.py**

```python
import asyncio
import uuid


class SSEManager:
    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}

    def create(self) -> str:
        task_id = str(uuid.uuid4())[:8]
        self._queues[task_id] = asyncio.Queue()
        return task_id

    async def push(self, task_id: str, event: str, data: str):
        if task_id in self._queues:
            await self._queues[task_id].put({"event": event, "data": data})

    async def close(self, task_id: str):
        await self.push(task_id, "done", "")
        await asyncio.sleep(0.5)
        self._queues.pop(task_id, None)

    async def events(self, task_id: str):
        q = self._queues.get(task_id)
        if not q:
            yield {"event": "error", "data": "<p>任务不存在</p>"}
            return
        while True:
            msg = await q.get()
            yield msg
            if msg["event"] == "done":
                break


sse_manager = SSEManager()
```

- [ ] **Step 2: 重写 main.py**

```python
import asyncio
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from config import settings
from memory import memory
from orchestrator import Orchestrator
from reviewer import Judge, Integrator
from sse_manager import sse_manager

app = FastAPI(title="AI 军团")
templates = Jinja2Templates(directory="templates")

orchestrator = Orchestrator()
judge = Judge()
integrator = Integrator()

DEFAULT_SESSION = "default"


async def process_pipeline(task_id: str, user_message: str, session_id: str):
    async def push(event, data):
        await sse_manager.push(task_id, event, data)

    try:
        await push("thinking", '<div class="status-card"><span class="spinner"></span> L1 总控正在分析任务...</div>')

        plan = await orchestrator.analyze(user_message)
        await push("thinking", f'<div class="status-card">任务类型: {plan.get("task_type")} | 能力: {", ".join(plan.get("capabilities", []))} | 候选数: {plan.get("candidates")}</div>')

        context = [
            {"role": m["role"], "content": m["content"]}
            for m in memory.get_history(session_id, last_n=settings.MAX_HISTORY_MESSAGES)
        ]

        agent_results = await orchestrator.dispatch(user_message, plan, context, push)

        await push("thinking", '<div class="status-card">🏛️ L3 评审团正在打分...</div>')

        evaluation = await judge.evaluate(agent_results)
        winner_name = evaluation.get("winner", agent_results[0]["agent"])
        winner_content = next((r["content"] for r in agent_results if r["agent"] == winner_name), agent_results[0]["content"])
        highlights = evaluation.get("highlights_from_others", [])

        await push("thinking", f'<div class="status-card">📊 最优方案来自：「{winner_name}」，正在整合...</div>')

        final = await integrator.integrate(user_message, winner_name, winner_content, highlights)

        memory.add(session_id, "assistant", final)

        review_html = f"""
        <div class="review-card">
          <h4>📊 评审结果 — 最优方案：「{winner_name}」</h4>
          <div class="review-detail">{evaluation.get('winner_reason', '')}</div>
        </div>
        """
        await push("review", review_html)

        await push("final", f'<div class="message assistant">{final}</div>')

    except Exception as e:
        error_html = f'<div class="error-card">处理出错: {str(e)}</div>'
        await push("final", error_html)
    finally:
        await sse_manager.close(task_id)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "session_id": DEFAULT_SESSION})


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "session_id": DEFAULT_SESSION})


@app.post("/chat", response_class=HTMLResponse)
async def chat_send(request: Request, message: str = Form(...)):
    session_id = DEFAULT_SESSION
    memory.add(session_id, "user", message)

    task_id = sse_manager.create()
    asyncio.create_task(process_pipeline(task_id, message, session_id))

    user_msg_html = f'<div class="message user">{message}</div>'
    sse_connect = f"""
    {user_msg_html}
    <div id="task-{task_id}" hx-ext="sse" sse-connect="/stream/{task_id}">
      <div sse-swap="thinking" id="thinking-{task_id}">
        <div class="status-card"><span class="spinner"></span> 正在思考...</div>
      </div>
      <div sse-swap="review" id="review-{task_id}"></div>
      <div sse-swap="final" id="final-{task_id}"></div>
    </div>
    """
    return HTMLResponse(user_msg_html + sse_connect)


@app.get("/stream/{task_id}")
async def stream(task_id: str, request: Request):
    async def event_gen():
        async for msg in sse_manager.events(task_id):
            yield {"event": msg["event"], "data": msg["data"]}
    return EventSourceResponse(event_gen())


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
```

- [ ] **Step 3: 验证路由注册**

```bash
cd agents_legion && python -c "
from main import app
routes = [r.path for r in app.routes]
assert '/' in routes
assert '/chat' in routes
assert '/stream/{task_id}' in routes
print('Routes OK:', routes)
"
```

---

### Task 9: HTML 模板（聊天界面）

**Files:**
- Create: `agents_legion/templates/base.html`
- Create: `agents_legion/templates/chat.html`

- [ ] **Step 1: 编写 base.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI 军团 — 多智能体协同</title>
  <script src="https://unpkg.com/htmx.org@2.0.4"></script>
  <script src="https://unpkg.com/htmx.org/dist/ext/sse.js"></script>
  <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
  <style>
    :root {
      --bg: #0f172a; --surface: #1e293b; --border: #334155;
      --text: #f1f5f9; --muted: #94a3b8; --accent: #6366f1;
      --green: #10b981; --yellow: #f59e0b; --red: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: var(--bg); color: var(--text); min-height: 100vh;
    }
    .container { max-width: 800px; margin: 0 auto; padding: 16px; }
    .header {
      text-align: center; padding: 20px 0 12px;
      border-bottom: 1px solid var(--border); margin-bottom: 16px;
    }
    .header h1 { font-size: 1.4em; color: var(--accent); }
    .header p { font-size: 0.85em; color: var(--muted); margin-top: 4px; }
    .chat-area { min-height: 60vh; max-height: 70vh; overflow-y: auto; padding: 8px 0; }
    .message {
      margin: 10px 0; padding: 12px 16px; border-radius: 10px;
      max-width: 90%; line-height: 1.7; white-space: pre-wrap; word-break: break-word;
    }
    .message.user { background: var(--accent); margin-left: auto; color: #fff; }
    .message.assistant { background: var(--surface); border: 1px solid var(--border); }
    .status-card {
      font-size: 0.82em; color: var(--muted); padding: 6px 12px;
      margin: 4px 0; border-left: 3px solid var(--accent);
      background: rgba(99,102,241,0.05); border-radius: 0 8px 8px 0;
    }
    .review-card {
      background: rgba(16,185,129,0.08); border: 1px solid var(--green);
      border-radius: 10px; padding: 12px 16px; margin: 8px 0; font-size: 0.85em;
    }
    .review-card h4 { color: var(--green); margin-bottom: 6px; }
    .error-card {
      background: rgba(239,68,68,0.1); border: 1px solid var(--red);
      border-radius: 8px; padding: 12px; color: var(--red); font-size: 0.85em;
    }
    .input-area { margin-top: 16px; display: flex; gap: 10px; }
    .input-area textarea {
      flex: 1; background: var(--surface); border: 1px solid var(--border);
      color: var(--text); padding: 12px; border-radius: 10px; resize: none;
      font-size: 0.95em; font-family: inherit; min-height: 50px;
    }
    .input-area textarea:focus { outline: none; border-color: var(--accent); }
    .input-area button {
      background: var(--accent); color: #fff; border: none;
      padding: 12px 24px; border-radius: 10px; cursor: pointer;
      font-size: 0.95em; font-weight: 600;
    }
    .input-area button:hover { opacity: 0.9; }
    .input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
    .spinner {
      display: inline-block; width: 14px; height: 14px;
      border: 2px solid var(--muted); border-top-color: var(--accent);
      border-radius: 50%; animation: spin 0.8s linear infinite;
      vertical-align: middle; margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    @media (max-width: 640px) {
      .container { padding: 10px; }
      .message { max-width: 95%; font-size: 0.9em; }
    }
  </style>
</head>
<body>
  <div class="container">
    {% block content %}{% endblock %}
  </div>
</body>
</html>
```

- [ ] **Step 2: 编写 chat.html**

```html
{% extends "base.html" %}
{% block content %}
<div class="header">
  <h1>AI 军团</h1>
  <p>三层多智能体协同 · 发散 + 严谨 + 批判</p>
</div>

<div id="chat-area" class="chat-area">
  <div class="message assistant">
    你好！我是 AI 军团，由三个智能体协同为你服务：
    🌊 发散型（广度探索）、🔬 严谨型（逻辑推演）、⚔️ 批判型（漏洞发现）。

    请输入你的问题，我会调动三路智能体并行分析，然后评审整合出最优答案。
  </div>
</div>

<form
  id="chat-form"
  hx-post="/chat"
  hx-target="#chat-area"
  hx-swap="beforeend"
  hx-on::after-request="this.reset(); document.querySelector('#send-btn').disabled = false"
  hx-on::before-request="document.querySelector('#send-btn').disabled = true"
>
  <div class="input-area">
    <textarea
      name="message"
      id="msg-input"
      rows="2"
      placeholder="输入你的问题..."
      required
    ></textarea>
    <button type="submit" id="send-btn">发送</button>
  </div>
</form>
{% endblock %}
```

- [ ] **Step 3: 启动服务并检查页面**

```bash
cd agents_legion && python main.py
```
打开 http://127.0.0.1:8000 检查：
- 聊天界面正确渲染
- 输入框和发送按钮可见
- 欢迎消息显示

---

### Task 10: 端到端验证

- [ ] **Step 1: 安装依赖**

```bash
cd agents_legion && pip install -r requirements.txt
```

- [ ] **Step 2: 设置环境变量**

创建 `.env` 文件：
```bash
cd agents_legion
cat > .env << 'ENVEOF'
ANTHROPIC_API_KEY=sk-ant-your-key-here
DEEPSEEK_API_KEY=sk-your-deepseek-key
ORCHESTRATOR_MODEL=claude-sonnet-4-6
AGENT_MODEL=claude-sonnet-4-6
REVIEWER_MODEL=claude-sonnet-4-6
ENVEOF
```

- [ ] **Step 3: 启动服务**

```bash
cd agents_legion && python main.py
```

- [ ] **Step 4: 发送一个测试任务**

在浏览器 http://127.0.0.1:8000 输入：

> "帮我分析：大三学生准备数学建模竞赛，应该选择A题（物理类）还是C题（数据分析类）？"

验证：
- 用户消息显示在聊天区
- L1 分析状态卡片出现
- 三个 Agent 的状态依次显示
- 评审卡片出现（显示最优方案）
- 最终答案流式渲染
- 移动端（Chrome DevTools 模拟）显示正常

- [ ] **Step 5: 提交**

```bash
git add agents_legion/ && git commit -m "feat: complete multi-agent pipeline with web chat UI"
```

---
