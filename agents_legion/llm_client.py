import time
import httpx
from openai import AsyncOpenAI
from config import settings

# 全局变量，main.py 在处理任务前设置，LLM 调用时自动上报
_active_task_id: str | None = None


def set_active_task(task_id: str | None):
    global _active_task_id
    _active_task_id = task_id


def _make_client(api_key: str, base_url: str) -> AsyncOpenAI:
    """创建禁用代理的 OpenAI 客户端，解决系统代理 SSL 问题"""
    http_client = httpx.AsyncClient(proxy=None, trust_env=False)
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
    )


async def call_qianwen(system_prompt: str, messages: list[dict], model: str | None = None) -> str:
    """调用 Qwen API（OpenAI 兼容接口）"""
    model = model or "qwen3.7-max"
    client = _make_client(settings.QWEN_API_KEY, settings.QWEN_BASE_URL)
    formatted = [{"role": "system", "content": system_prompt}]
    for m in messages:
        formatted.append({"role": m["role"], "content": m["content"]})
    start = time.time()
    response = await client.chat.completions.create(
        model=model,
        messages=formatted,
        max_tokens=65536,
        extra_body={"enable_thinking": True},
    )
    _record(model, response.usage, time.time() - start)
    return response.choices[0].message.content


async def call_deepseek(system_prompt: str, messages: list[dict], model: str | None = None) -> str:
    """调用 DeepSeek API（OpenAI 兼容接口）"""
    model = model or "deepseek-v4-pro"
    client = _make_client(settings.DEEPSEEK_API_KEY, settings.DEEPSEEK_BASE_URL)
    formatted = [{"role": "system", "content": system_prompt}]
    for m in messages:
        formatted.append({"role": m["role"], "content": m["content"]})
    start = time.time()
    response = await client.chat.completions.create(
        model=model,
        messages=formatted,
        max_tokens=65536,
        extra_body={"enable_thinking": True},
    )
    _record(model, response.usage, time.time() - start)
    return response.choices[0].message.content


async def call_glm(system_prompt: str, messages: list[dict], model: str | None = None) -> str:
    """调用 GLM API（OpenAI 兼容接口）"""
    model = model or "glm-5.1"
    client = _make_client(settings.GLM_API_KEY, settings.GLM_BASE_URL)
    formatted = [{"role": "system", "content": system_prompt}]
    for m in messages:
        formatted.append({"role": m["role"], "content": m["content"]})
    start = time.time()
    response = await client.chat.completions.create(
        model=model,
        messages=formatted,
        max_tokens=65536,
    )
    _record(model, response.usage, time.time() - start)
    return response.choices[0].message.content


def _record(model: str, usage, duration: float):
    """内部：如果设置了 active task，自动上报"""
    if _active_task_id and usage:
        from cost_tracker import cost_tracker
        cost_tracker.record(
            _active_task_id, model,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
            duration,
        )
