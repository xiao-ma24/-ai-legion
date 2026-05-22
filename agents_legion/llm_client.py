import httpx
from openai import AsyncOpenAI
from config import settings


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
    response = await client.chat.completions.create(
        model=model,
        messages=formatted,
        max_tokens=65536,
        extra_body={"enable_thinking": True},
    )
    return response.choices[0].message.content


async def call_deepseek(system_prompt: str, messages: list[dict], model: str | None = None) -> str:
    """调用 DeepSeek API（OpenAI 兼容接口）"""
    model = model or "deepseek-v4-pro"
    client = _make_client(settings.DEEPSEEK_API_KEY, settings.DEEPSEEK_BASE_URL)
    formatted = [{"role": "system", "content": system_prompt}]
    for m in messages:
        formatted.append({"role": m["role"], "content": m["content"]})
    response = await client.chat.completions.create(
        model=model,
        messages=formatted,
        max_tokens=65536,
        extra_body={"enable_thinking": True},
    )
    return response.choices[0].message.content


async def call_glm(system_prompt: str, messages: list[dict], model: str | None = None) -> str:
    """调用 GLM API（OpenAI 兼容接口）"""
    model = model or "glm-5.1"
    client = _make_client(settings.GLM_API_KEY, settings.GLM_BASE_URL)
    formatted = [{"role": "system", "content": system_prompt}]
    for m in messages:
        formatted.append({"role": m["role"], "content": m["content"]})
    response = await client.chat.completions.create(
        model=model,
        messages=formatted,
        max_tokens=65536,
    )
    return response.choices[0].message.content
