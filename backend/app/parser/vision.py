import base64
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

VISION_PROMPT = (
    "You are analyzing images extracted from a single page of a financial document. "
    "Describe what each image shows in detail, including all visible data, values, trends, labels, and axes. "
    "If an image is a chart or graph, describe the data it represents numerically where possible. "
    "If an image is a table, transcribe it. "
    "If an image is decorative or a logo, say so briefly. "
    "Be concise but complete."
)


def describe_images(images_bytes: list[tuple[bytes, str]]) -> str | None:
    """Send all images from a page to the vision LLM in one call. Returns a combined description or None on failure."""
    if not images_bytes:
        return None
    try:
        if settings.LLM_PROVIDER == "openai":
            return _describe_openai(images_bytes)
        elif settings.LLM_PROVIDER == "anthropic":
            return _describe_anthropic(images_bytes)
        else:
            logger.warning(f"Unknown LLM_PROVIDER '{settings.LLM_PROVIDER}' for vision — skipping images")
            return None
    except Exception as e:
        logger.warning(f"Vision LLM call failed: {e}")
        return None


def _b64(image_bytes: bytes) -> str:
    return base64.standard_b64encode(image_bytes).decode("utf-8")


def _describe_openai(images_bytes: list[tuple[bytes, str]]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    content = [{"type": "text", "text": VISION_PROMPT}]
    for img_bytes, media_type in images_bytes:
        content.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{_b64(img_bytes)}"}})

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def _describe_anthropic(images_bytes: list[tuple[bytes, str]]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    content = []
    for img_bytes, media_type in images_bytes:
        content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": _b64(img_bytes)}})
    content.append({"type": "text", "text": VISION_PROMPT})

    response = client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text.strip()
