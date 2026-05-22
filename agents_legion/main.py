import asyncio
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from config import settings
from memory import memory
from orchestrator import Orchestrator
from reviewer import Judge, Integrator
from task_manager import task_manager

BASE_DIR = Path(__file__).parent
app = FastAPI(title="AI 军团")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

orchestrator = Orchestrator()
judge = Judge()
integrator = Integrator()


def get_session(request: Request) -> str:
    sid = request.cookies.get("ai_legion_session", "")
    if sid and memory.exists(sid):
        return sid
    return memory.create()


def render_page(request: Request, session_id: str) -> HTMLResponse:
    sessions = memory.list_sessions()
    history = memory.get_history(session_id)
    template = templates.get_template("chat.html")
    html = template.render({
        "request": request,
        "session_id": session_id,
        "sessions": sessions,
        "history": history,
    })
    return HTMLResponse(html)


async def process_pipeline(task_id: str, user_message: str, session_id: str):
    try:
        # ── L1 分析 ──
        task_manager.push(task_id, _stage_card("L1", "正在评估任务复杂度...", "active"))
        plan = await orchestrator.analyze(user_message)
        complexity = plan.get("complexity", "analytical")
        caps = ", ".join(plan.get("capabilities", []))

        task_manager.push(task_id, _stage_card(
            "L1",
            f"「{orchestrator.complexity_label(complexity)}」能力: {caps}",
            "done",
        ))

        context = [
            {"role": m["role"], "content": m["content"]}
            for m in memory.get_history(session_id, last_n=settings.MAX_HISTORY_MESSAGES)
        ]

        # ── L2 执行（按复杂度路由） ──
        if complexity == "simple":
            await _run_simple(task_id, user_message, context, plan, session_id)

        elif complexity == "analytical":
            await _run_analytical(task_id, user_message, context, plan, session_id)

        else:  # strategic
            await _run_strategic(task_id, user_message, context, plan, session_id)

    except Exception as e:
        task_manager.set_final(task_id, f'<div class="error-card">处理出错: {str(e)}</div>')
        task_manager.mark_done(task_id)


# ── 三种路径 ──────────────────────────────────

async def _run_simple(task_id, user_message, context, plan, session_id):
    task_manager.push(task_id, _stage_card("L2", "严谨型正在分析...", "active"))
    agent_results, _ = await orchestrator.dispatch(user_message, plan, context)
    final = agent_results[0]["content"]
    memory.add(session_id, "assistant", final)

    task_manager.push(task_id, _stage_card("L2", "严谨型分析完成", "done"))
    final_html = _final_result("simple", "严谨型", final)
    task_manager.set_final(task_id, final_html)
    task_manager.mark_done(task_id)


async def _run_analytical(task_id, user_message, context, plan, session_id):
    task_manager.push(task_id, _stage_card("L2", "发散型 + 严谨型 正在并行分析...", "active"))
    agent_results, _ = await orchestrator.dispatch(user_message, plan, context)

    for r in agent_results:
        task_manager.push(task_id, _agent_card(r["agent"], r["content"]))

    # 一致性检查
    task_manager.push(task_id, _stage_card("L2", "正在检查两方案一致性...", "active"))
    consistency = await orchestrator.check_consistency(
        user_message,
        agent_results[0]["content"],
        agent_results[1]["content"],
    )

    if consistency.get("consistent") and consistency.get("confidence", 0) >= 0.85:
        # Early Exit — 直接选更好的一份
        better = agent_results[consistency["better_index"]]["content"]
        winner_name = agent_results[consistency["better_index"]]["agent"]
        task_manager.push(task_id, _stage_card(
            "L2", f"两方案高度一致 (置信度 {consistency['confidence']:.0%})，跳过评审", "done"
        ))
        memory.add(session_id, "assistant", better)
        final_html = _final_result("analytical", winner_name, better)
    else:
        # 走 Judge
        task_manager.push(task_id, _stage_card("L3", "方案存在分歧，评审团介入...", "active"))
        evaluation = await judge.evaluate(agent_results)
        winner_name = evaluation.get("winner", agent_results[0]["agent"])
        winner_content = next(
            (r["content"] for r in agent_results if r["agent"] == winner_name),
            agent_results[0]["content"],
        )
        task_manager.push(task_id, _stage_card(
            "L3", f"评审完成 — 最优方案: 「{winner_name}」", "done"
        ))
        memory.add(session_id, "assistant", winner_content)
        final_html = _final_result("analytical", winner_name, winner_content)

    task_manager.set_final(task_id, final_html)
    task_manager.mark_done(task_id)


async def _run_strategic(task_id, user_message, context, plan, session_id):
    task_manager.push(task_id, _stage_card("L2", "发散型 + 严谨型 + 批判型 三智能体全链路启动...", "active"))
    agent_results, _ = await orchestrator.dispatch(user_message, plan, context)

    for r in agent_results:
        task_manager.push(task_id, _agent_card(r["agent"], r["content"]))

    task_manager.push(task_id, _stage_card("L3", "评审团 7维度打分中...", "active"))
    evaluation = await judge.evaluate(agent_results)

    winner_name = evaluation.get("winner", agent_results[0]["agent"])
    winner_content = next(
        (r["content"] for r in agent_results if r["agent"] == winner_name),
        agent_results[0]["content"],
    )
    highlights = evaluation.get("highlights_from_others", [])

    task_manager.push(task_id, _stage_card(
        "L3", f"最优方案: 「{winner_name}」— {evaluation.get('winner_reason', '')}", "done"
    ))

    task_manager.push(task_id, _stage_card("L3", "整合器正在融合最终答案...", "active"))
    final = await integrator.integrate(user_message, winner_name, winner_content, highlights)
    memory.add(session_id, "assistant", final)

    final_html = _final_result("strategic", winner_name, final)
    task_manager.set_final(task_id, final_html)
    task_manager.mark_done(task_id)


# ── HTML 片段 ──────────────────────────────────

def _stage_card(layer: str, text: str, state: str) -> str:
    num = {"L1": "1", "L2": "2", "L3": "3"}.get(layer, "?")
    done_class = "stage-done" if state == "done" else ""
    spinner = "" if state == "done" else '<span class="spinner"></span>'
    return f"""<div class="stage-card {done_class}">
      <span class="stage-num">{num}</span>{spinner}
      <span class="stage-text">{text}</span></div>"""


def _agent_card(name: str, content: str) -> str:
    emoji = {"发散型": "🌊", "严谨型": "🔬", "批判型": "⚔️"}.get(name, "🤖")
    short = content[:120].replace("\n", " ").replace('"', "&quot;")
    return f"""<div class="agent-card">
      <div class="agent-card-header" onclick="this.parentElement.classList.toggle('expanded')">
        <span>{emoji} {name}</span><span class="agent-card-arrow">▾</span>
      </div>
      <div class="agent-card-preview">{short}...</div>
      <div class="agent-card-full">{content.replace(chr(10), '<br>')}</div></div>"""


def _final_result(level: str, source: str, content: str) -> str:
    labels = {"simple": ("🟢", "简单任务完成"), "analytical": ("🟡", "双智能体协同完成"), "strategic": ("🔴", "三智能体全链路完成")}
    icon, label = labels.get(level, ("●", ""))
    return f"""<div class="result-summary">{icon} {label} — 来源: {source}</div>
<div class="message assistant">{content}</div>"""


# ── 路由 ──────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse("/chat")


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    session_id = get_session(request)
    resp = render_page(request, session_id)
    resp.set_cookie("ai_legion_session", session_id)
    return resp


@app.get("/chat/{session_id}", response_class=HTMLResponse)
async def chat_switch(request: Request, session_id: str):
    if not memory.exists(session_id):
        return RedirectResponse("/chat")
    resp = render_page(request, session_id)
    resp.set_cookie("ai_legion_session", session_id)
    return resp


@app.post("/chat/new")
async def chat_new(request: Request):
    session_id = memory.create()
    resp = RedirectResponse(f"/chat/{session_id}", status_code=303)
    resp.set_cookie("ai_legion_session", session_id)
    return resp


@app.post("/chat/{session_id}/delete")
async def chat_delete(request: Request, session_id: str):
    memory.clear(session_id)
    return RedirectResponse("/chat", status_code=303)


@app.post("/chat", response_class=HTMLResponse)
async def chat_send(request: Request, message: str = Form(...)):
    sid = request.cookies.get("ai_legion_session", "")
    if not sid or not memory.exists(sid):
        sid = memory.create()

    memory.add(sid, "user", message)
    task_id = task_manager.create()
    asyncio.create_task(process_pipeline(task_id, message, sid))

    user_html = f'<div class="message user">{message}</div>'
    poll_html = f"""{user_html}
<div id="task-{task_id}" hx-get="/poll/{task_id}" hx-trigger="every 1.5s" hx-swap="outerHTML">
  <div class="stage-card"><span class="spinner"></span><span class="stage-text">AI 军团正在启动...</span></div>
</div>"""
    resp = HTMLResponse(poll_html)
    resp.set_cookie("ai_legion_session", sid)
    return resp


@app.get("/poll/{task_id}", response_class=HTMLResponse)
async def poll(task_id: str, request: Request):
    if task_manager.is_done(task_id):
        html = task_manager.get_final(task_id)
        sidebar = _render_sidebar()
        return HTMLResponse(html + sidebar)
    else:
        progress = task_manager.get_progress(task_id)
        return HTMLResponse(f"""
<div id="task-{task_id}" hx-get="/poll/{task_id}" hx-trigger="every 1.5s" hx-swap="outerHTML">
  {progress}
</div>""")


def _render_sidebar() -> str:
    sessions = memory.list_sessions()
    items = []
    for s in sessions:
        items.append(
            f'<a href="/chat/{s["id"]}" class="sess-item"><span class="sess-title">{s["title"]}</span>'
            f'<span class="sess-meta">{s["count"]}条 · {s["updated"]}</span></a>'
        )
    return f"""<div id="sidebar" hx-swap-oob="outerHTML">
  <div class="sidebar-header">
    <h2>AI 军团</h2>
    <form method="post" action="/chat/new"><button class="btn-new">+ 新对话</button></form>
  </div>
  <div class="session-list">{''.join(items)}</div>
</div>"""


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
