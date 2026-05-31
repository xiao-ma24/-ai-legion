import asyncio
import logging
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


from summary_manager import summary_manager
from intent_classifier import intent_classifier
from routing_policy import routing_policy
from llm_client import call_qianwen, set_active_task
from cost_tracker import cost_tracker as cost_log

from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
app = FastAPI(title="AI 军团")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

orchestrator = Orchestrator()
judge = Judge()
integrator = Integrator()

# ── 日志 ──
logging.basicConfig(
    filename=str(BASE_DIR / "data" / "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger("pipeline")

LIGHTWEIGHT_PROMPT = """你是一个智能助手的快速回复模式。
用户发送了一条简短消息（确认、闲聊、或澄清），请给出简洁、自然的回应。
如果有对话上下文，请基于上下文回复。回复控制在100字以内。"""


def get_session(request: Request) -> str | None:
    """获取当前会话 ID，没有则返回 None（不自动创建）"""
    sid = request.cookies.get("ai_legion_session", "")
    if sid and memory.exists(sid):
        return sid
    return None


def render_page(request: Request, session_id: str | None = None) -> HTMLResponse:
    sessions = memory.list_sessions()
    # 清理空会话
    for s in list(sessions):
        if s["count"] == 0 and s["id"] != session_id:
            memory.clear(s["id"])
    sessions = memory.list_sessions()
    history = memory.get_history(session_id) if session_id else []
    template = templates.get_template("chat.html")
    html = template.render({
        "request": request,
        "session_id": session_id or "",
        "sessions": sessions,
        "history": history,
    })
    return HTMLResponse(html)


# ══════════════════════════════════════════════════
# 核心流程
# ══════════════════════════════════════════════════

async def process_pipeline(task_id: str, user_message: str, session_id: str):
    set_active_task(task_id)  # 全局激活，LLM 调用自动上报费用
    try:
        # ── 获取会话摘要 ──
        summary = summary_manager.get(session_id)
        session_ctx = summary.to_context_string()

        # ── 意图分类 ──
        _push_stage(task_id, "L1", "L1", "正在理解你的意图...", "active")

        intent_result = await intent_classifier.classify(
            user_message,
            session_topic=summary.current_topic,
            depth=summary.depth,
        )
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        intent_label = orchestrator.intent_label(intent)
        _push_stage(task_id, "L1", "L1",
            f"意图: {intent_label} (置信度 {confidence:.0%})", "done")

        logger.info(
            f"[{session_id}] intent={intent} conf={confidence:.2f} "
            f"depth={summary.depth} topic={summary.current_topic}"
        )

        # ── 低置信度 → 保守策略 ──
        if confidence < 0.75:
            intent = "followup"
            confidence = 0.75
            _push_stage(task_id, "L1", "L1", "意图不确定，采用保守策略（追问深入）", "done")

        # ── 路由 ──
        if intent == "casual_confirm" or intent == "clarify":
            final = await _run_lightweight(task_id, user_message, session_ctx, session_id, intent)
        elif intent == "new_topic":
            final = await _run_new_topic(task_id, user_message, session_id, session_ctx, intent)
        elif intent == "followup":
            final = await _run_followup(task_id, user_message, session_id, session_ctx, intent)
        elif intent == "compare_choice":
            final = await _run_compare(task_id, user_message, session_id, session_ctx, intent)
        elif intent == "execute":
            final = await _run_execute(task_id, user_message, session_id, session_ctx, intent)
        else:
            final = await _run_followup(task_id, user_message, session_id, session_ctx, intent)

        # ── 更新摘要 ──
        await summary_manager.update(session_id, user_message, final, intent)

        task_manager.mark_done(task_id)

    except Exception as e:
        logger.error(f"[{session_id}] ERROR: {e}")
        task_manager.set_final(task_id, f'<div class="error-card">处理出错: {str(e)}</div>')
        task_manager.mark_done(task_id)
    finally:
        set_active_task(None)  # 清理，避免泄漏到其他任务


# ══════════════════════════════════════════════════
# 各路由路径
# ══════════════════════════════════════════════════

async def _run_lightweight(task_id, message, session_ctx, session_id, intent) -> str:
    """轻量回复：不进入 Agent Pipeline"""
    _push_stage(task_id, "L2", "L2", "快速回复中...", "active")

    ctx_block = f"\n对话上下文:\n{session_ctx}\n" if session_ctx else ""
    result = await call_qianwen(
        LIGHTWEIGHT_PROMPT + ctx_block,
        [{"role": "user", "content": message}],
    )

    memory.add(session_id, "assistant", result)
    _push_stage(task_id, "L2", "L2", "快速回复完成", "done")

    html = _final_result("lightweight", result, task_id)
    task_manager.set_final(task_id, html)
    return result


async def _run_new_topic(task_id, message, session_id, session_ctx, intent) -> str:
    """新话题：完整复杂度分析 → simple/analytical/strategic"""
    plan = await orchestrator.analyze(message)
    complexity = plan.get("complexity", "analytical")
    caps = plan.get("capabilities", [])

    _push_stage(task_id, "L1", "L1",
        f"「{orchestrator.complexity_label(complexity)}」能力: {', '.join(caps)}", "done")

    # 预搜索
    async def push_tool(layer, text, state):
        _push_stage(task_id, "L1_tool", layer, text, state)

    context_data = await orchestrator.prepare_context(message, caps, push_fn=push_tool)
    search_ctx = orchestrator.format_context_for_agent(context_data)

    # 显示搜索结果摘要卡片（不管成功失败都显示，让用户知道工具启动了）
    web_results = context_data.get("search_results", [])
    arxiv_results = context_data.get("arxiv_results", [])
    search_html = ""
    if web_results:
        search_html += _search_results_card("🔍 网页搜索", web_results)
    if arxiv_results:
        search_html += _search_results_card("📄 ArXiv 论文", arxiv_results)
    task_manager.push(task_id, search_html)

    context = _build_context(session_id)

    if complexity == "simple":
        return await _exec_simple(task_id, message, context, caps, session_id, session_ctx, intent, search_ctx, search_html)
    elif complexity == "analytical":
        return await _exec_analytical(task_id, message, context, caps, session_id, session_ctx, intent, search_ctx, search_html)
    else:
        return await _exec_strategic(task_id, message, context, caps, session_id, session_ctx, intent, search_ctx, search_html)


async def _run_followup(task_id, message, session_id, session_ctx, intent) -> str:
    """追问深入：单 Agent 或 双 Agent（含风险时）"""
    roles, strategy = routing_policy.get_roles(intent, 0.85, ["research"], message)
    # 排除 full_pipeline（followup 不应走完整链路）
    roles = [r for r in roles if r != "full_pipeline"] or ["rigorous"]

    caps = ["research"]
    context = _build_context(session_id)

    # 追问也需要搜索
    async def push_tool(layer, text, state):
        _push_stage(task_id, "L1_tool", layer, text, state)
    context_data = await orchestrator.prepare_context(message, caps, push_fn=push_tool)
    search_ctx = orchestrator.format_context_for_agent(context_data)

    web_results = context_data.get("search_results", [])
    arxiv_results = context_data.get("arxiv_results", [])
    search_html = ""
    if web_results:
        search_html += _search_results_card("🔍 网页搜索", web_results)
    if arxiv_results:
        search_html += _search_results_card("📄 ArXiv 论文", arxiv_results)

    label = " + ".join([{"rigorous": "严谨型", "divergent": "发散型", "critical": "批判型"}.get(r, r) for r in roles])
    _push_stage(task_id, "L2", "L2", f"追问模式 → {label} 深入分析中...", "active")

    agent_results = await orchestrator.dispatch(
        message, roles, caps, context, session_ctx, intent, search_ctx
    )

    for r in agent_results:
        task_manager.push(task_id, _agent_card(r["agent"], r["content"]))

    # 选最佳输出
    final = agent_results[0]["content"]
    winner = agent_results[0]["agent"]

    if len(agent_results) > 1:
        evaluation = await judge.evaluate(agent_results, session_ctx)
        winner = evaluation.get("winner", agent_results[0]["agent"])
        final = next(
            (r["content"] for r in agent_results if r["agent"] == winner),
            agent_results[0]["content"],
        )
        _push_stage(task_id, "L3_judge", "L3", f"追问评审 → 最优: 「{winner}」", "done")

    memory.add(session_id, "assistant", final)
    _push_stage(task_id, "L2", "L2", f"{label} 深入分析完成", "done")
    html = _final_result("followup", final, task_id, search_html)
    task_manager.set_final(task_id, html)
    return final


async def _run_compare(task_id, message, session_id, session_ctx, intent) -> str:
    """对比选择：发散 + 严谨 + Judge"""
    roles, strategy = routing_policy.get_roles(intent, 0.85, ["research", "writing"], message)
    roles = [r for r in roles if r != "full_pipeline"] or ["divergent", "rigorous"]

    caps = ["research", "writing"]
    context = _build_context(session_id)

    _push_stage(task_id, "L2", "L2", "对比模式 → 发散型 + 严谨型 并行分析...", "active")

    agent_results = await orchestrator.dispatch(
        message, roles, caps, context, session_ctx, intent, ""
    )

    for r in agent_results:
        task_manager.push(task_id, _agent_card(r["agent"], r["content"]))

    _push_stage(task_id, "L3_judge", "L3", "对比评审中...", "active")
    evaluation = await judge.evaluate(agent_results, session_ctx)
    winner = evaluation.get("winner", agent_results[0]["agent"])
    final = next(
        (r["content"] for r in agent_results if r["agent"] == winner),
        agent_results[0]["content"],
    )

    _push_stage(task_id, "L3_judge", "L3", f"对比完成 → 推荐: 「{winner}」", "done")
    memory.add(session_id, "assistant", final)
    html = _final_result("compare", final, task_id)
    task_manager.set_final(task_id, html)
    return final


async def _run_execute(task_id, message, session_id, session_ctx, intent) -> str:
    """执行落地：单 Agent"""
    # 简单分析能力需求
    plan = await orchestrator.analyze(message)
    caps = plan.get("capabilities", ["writing"])

    roles, strategy = routing_policy.get_roles(intent, 0.85, caps, message)
    roles = [r for r in roles if r != "full_pipeline"] or ["rigorous"]

    context = _build_context(session_id)

    _push_stage(task_id, "L2", "L2", "执行模式 → 严谨型 生成中...", "active")

    agent_results = await orchestrator.dispatch(
        message, roles, caps, context, session_ctx, intent, ""
    )

    final = agent_results[0]["content"] if agent_results else "执行未返回结果"
    memory.add(session_id, "assistant", final)
    _push_stage(task_id, "L2", "L2", "执行完成", "done")
    html = _final_result("execute", final, task_id)
    task_manager.set_final(task_id, html)
    return final


# ── 复杂度子路径（new_topic 内部调用） ──

async def _exec_simple(task_id, message, context, caps, session_id, session_ctx, intent, search_ctx="", search_html="") -> str:
    _push_stage(task_id, "L2", "L2", "严谨型正在分析...", "active")
    results = await orchestrator.dispatch(message, ["rigorous"], caps, context, session_ctx, intent, search_ctx)
    final = results[0]["content"]
    memory.add(session_id, "assistant", final)
    _push_stage(task_id, "L2", "L2", "分析完成", "done")
    html = _final_result("simple", final, task_id, search_html)
    task_manager.set_final(task_id, html)
    return final


async def _exec_analytical(task_id, message, context, caps, session_id, session_ctx, intent, search_ctx="", search_html="") -> str:
    _push_stage(task_id, "L2", "L2", "发散型 + 严谨型 并行分析...", "active")
    results = await orchestrator.dispatch(message, ["divergent", "rigorous"], caps, context, session_ctx, intent, search_ctx)

    for r in results:
        task_manager.push(task_id, _agent_card(r["agent"], r["content"]))

    _push_stage(task_id, "L2_check", "L2", "一致性检查中...", "active")
    consistency = await orchestrator.check_consistency(message, results[0]["content"], results[1]["content"])

    if consistency.get("consistent") and consistency.get("confidence", 0) >= 0.85:
        idx = consistency["better_index"]
        final = results[idx]["content"]
        winner = results[idx]["agent"]
        _push_stage(task_id, "L2_check", "L2",
            f"高度一致 ({consistency['confidence']:.0%})，跳过评审", "done")
    else:
        _push_stage(task_id, "L3_judge", "L3", "方案有分歧，评审团介入...", "active")
        evaluation = await judge.evaluate(results, session_ctx)
        winner = evaluation.get("winner", results[0]["agent"])
        final = next((r["content"] for r in results if r["agent"] == winner), results[0]["content"])
        _push_stage(task_id, "L3_judge", "L3", f"评审完成 → 最优: 「{winner}」", "done")

    memory.add(session_id, "assistant", final)
    html = _final_result("analytical", final, task_id, search_html)
    task_manager.set_final(task_id, html)
    return final


async def _exec_strategic(task_id, message, context, caps, session_id, session_ctx, intent, search_ctx="", search_html="") -> str:
    _push_stage(task_id, "L2", "L2", "三智能体全链路启动...", "active")
    results = await orchestrator.dispatch(
        message, ["divergent", "rigorous", "critical"], caps, context, session_ctx, intent, search_ctx
    )

    for r in results:
        task_manager.push(task_id, _agent_card(r["agent"], r["content"]))

    _push_stage(task_id, "L3_judge", "L3", "评审团 7维度打分中...", "active")
    evaluation = await judge.evaluate(results, session_ctx)
    winner = evaluation.get("winner", results[0]["agent"])
    winner_content = next((r["content"] for r in results if r["agent"] == winner), results[0]["content"])
    highlights = evaluation.get("highlights_from_others", [])

    _push_stage(task_id, "L3_judge", "L3", f"最优: 「{winner}」", "done")
    _push_stage(task_id, "L3_integrate", "L3", "整合器融合中...", "active")

    final = await integrator.integrate(message, winner, winner_content, highlights, session_ctx)
    memory.add(session_id, "assistant", final)

    html = _final_result("strategic", final, task_id, search_html)
    task_manager.set_final(task_id, html)
    return final


# ══════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════

def _build_context(session_id: str) -> list[dict]:
    return [
        {"role": m["role"], "content": m["content"]}
        for m in memory.get_history(session_id, last_n=settings.MAX_HISTORY_MESSAGES)
    ]


def _stage_card(layer: str, text: str, state: str) -> str:
    num = {"L1": "1", "L2": "2", "L3": "3"}.get(layer, "🔧")
    done_class = "stage-done" if state == "done" else ""
    dot = "" if state == "done" else '<span class="pulse-dot"></span>'
    return f"""<div class="stage-card {done_class}">
      <span class="stage-num">{num}</span>{dot}
      <span class="stage-text">{text}</span></div>"""


def _push_stage(task_id: str, stage_key: str, layer: str, text: str, state: str):
    """推送阶段卡片，同 key 替换而非追加"""
    task_manager.push_stage(task_id, _stage_card(layer, text, state), stage_key)


def _search_results_card(title: str, results: list) -> str:
    if not results:
        return f"""<div class="search-results-card">
          <div class="sr-header">{title}</div>
          <ul class="sr-list"><li class="sr-item sr-empty">未获取到结果</li></ul>
        </div>"""
    items = []
    for r in results[:3]:
        if r.get("title") == "搜索暂无结果":
            items.append(f'<li class="sr-item sr-empty">{r["snippet"]}</li>')
            continue
        link = f'<a href="{r["url"]}" target="_blank" class="sr-link">{r["title"]}</a>' if r.get("url") else f'<span class="sr-title">{r.get("title", "")}</span>'
        snippet = r.get("snippet", r.get("summary", ""))[:80]
        authors = ""
        if r.get("authors"):
            authors = f' · {", ".join(r["authors"][:2])}'
        items.append(f'<li class="sr-item">{link}{authors}<br><span class="sr-snippet">{snippet}...</span></li>')
    return f"""<div class="search-results-card">
      <div class="sr-header">{title} ({len(results)}条)</div>
      <ul class="sr-list">{''.join(items)}</ul>
    </div>"""


def _agent_card(name: str, content: str) -> str:
    emoji = {"发散型": "🌊", "严谨型": "🔬", "批判型": "⚔️"}.get(name, "🤖")
    role_class = {"发散型": "role-divergent", "严谨型": "role-rigorous", "批判型": "role-critical"}.get(name, "")
    short = content[:120].replace("\n", " ").replace('"', "&quot;")
    return f"""<div class="agent-card">
      <div class="agent-card-header" onclick="this.parentElement.classList.toggle('expanded')">
        <span>{emoji} <span class="{role_class}">{name}</span></span><span class="agent-card-arrow">▾</span>
      </div>
      <div class="agent-card-preview">{short}...</div>
      <div class="agent-card-full">{content.replace(chr(10), '<br>')}</div></div>"""


def _final_result(level: str, content: str, task_id: str = "", extra_html: str = "") -> str:
    labels = {
        "lightweight": ("💬", "快速回复"),
        "simple": ("🟢", "简单任务完成"),
        "analytical": ("🟡", "双智能体协同完成"),
        "strategic": ("🔴", "三智能体全链路完成"),
        "followup": ("🔵", "追问深入完成"),
        "compare": ("🟠", "对比分析完成"),
        "execute": ("🟣", "执行完成"),
    }
    icon, label = labels.get(level, ("●", "完成"))
    cost_html = ""
    if task_id:
        summary = cost_log.get_summary(task_id)
        if summary:
            cost_html = (
                f'<div class="cost-line">'
                f'{summary["calls"]}次调用 · {summary["total_tokens"]:,} tokens · '
                f'¥{summary["total_cost"]:.4f}'
                f'</div>'
            )
    return f"""{extra_html}
<div class="result-summary">{icon} {label}</div>
<div class="message-row assistant">
  <div class="msg-avatar assistant-av">A</div>
  <div class="message assistant">{content}</div>
</div>
{cost_html}"""


# ══════════════════════════════════════════════════
# 路由
# ══════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse("/chat")


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    session_id = get_session(request)
    return render_page(request, session_id)


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
    summary_manager.reset(session_id)
    return RedirectResponse("/chat", status_code=303)


@app.post("/chat", response_class=HTMLResponse)
async def chat_send(request: Request, message: str = Form(...)):
    sid = request.cookies.get("ai_legion_session", "")
    if not sid or not memory.exists(sid):
        sid = memory.create()

    memory.add(sid, "user", message)
    task_id = task_manager.create()
    asyncio.create_task(process_pipeline(task_id, message, sid))

    user_html = f'<div class="message-row user"><div class="msg-avatar user-av">U</div><div class="message user">{message}</div></div>'
    poll_html = f"""{user_html}
<div id="task-{task_id}" hx-get="/poll/{task_id}" hx-trigger="every 1.5s" hx-swap="outerHTML">
  <div class="stage-card"><span class="pulse-dot"></span><span class="stage-text">AI 军团正在启动...</span></div>
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
            f'<div class="sess-item-wrap">'
            f'<a href="/chat/{s["id"]}" class="sess-item"><span class="sess-title">{s["title"]}</span>'
            f'<span class="sess-meta">{s["count"]}条 · {s["updated"]}</span></a>'
            f'<form method="post" action="/chat/{s["id"]}/delete" class="sess-delete-form">'
            f'<button type="submit" class="sess-delete" title="删除">✕</button></form></div>'
        )
    return f"""<div id="sidebar" hx-swap-oob="outerHTML">
  <div class="sidebar-header">
    <div class="sidebar-logo">
      <div class="logo-icon">A</div>
      <div><h2>AI LEGION</h2><span class="logo-sub">多智能体协同系统</span></div>
    </div>
    <form method="post" action="/chat/new"><button class="btn-new">✦ 新对话</button></form>
  </div>
  <div class="session-list">{''.join(items)}</div>
</div>"""


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
