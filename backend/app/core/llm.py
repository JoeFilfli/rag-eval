from app.core.config import settings


def get_llm_client():
    if settings.LLM_PROVIDER == "openai":
        return _openai_chat
    elif settings.LLM_PROVIDER == "anthropic":
        return _anthropic_chat
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")


def get_llm_client_with_usage():
    if settings.LLM_PROVIDER == "openai":
        return _openai_chat_with_usage
    elif settings.LLM_PROVIDER == "anthropic":
        return _anthropic_chat_with_usage
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")


def _openai_chat(messages: list[dict]) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(model=settings.LLM_MODEL, messages=messages)
    return response.choices[0].message.content.strip()


def _openai_chat_with_usage(messages: list[dict]) -> tuple[str, dict]:
    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(model=settings.LLM_MODEL, messages=messages)
    text = response.choices[0].message.content.strip()
    usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
    return text, usage


def _anthropic_chat(messages: list[dict]) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    user_messages = [m for m in messages if m["role"] != "system"]
    kwargs = {"model": settings.LLM_MODEL, "max_tokens": 2048, "messages": user_messages}
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text.strip()


def _anthropic_chat_with_usage(messages: list[dict]) -> tuple[str, dict]:
    import anthropic
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    user_messages = [m for m in messages if m["role"] != "system"]
    kwargs = {"model": settings.LLM_MODEL, "max_tokens": 2048, "messages": user_messages}
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    text = response.content[0].text.strip()
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    return text, usage
