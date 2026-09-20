"""Chat + vision calls across OpenAI-compatible, Anthropic and Google dialects."""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request

try:
    from .providers import (
        Provider,
        get_provider,
        is_chat_model,
        prettiest_name,
        resolve_key,
    )
except ImportError:
    from providers import (
        Provider,
        get_provider,
        is_chat_model,
        prettiest_name,
        resolve_key,
    )

TIMEOUT = 300
MAX_IMAGE_SIDE = 1024

REASONING_LEVELS = ("provider default", "minimal", "low", "medium", "high")


class LLMError(RuntimeError):
    pass


def _post(url: str, payload: dict, headers: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as res:
            return json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = ""
        try:
            detail = err.read().decode("utf-8")[:600]
        except Exception:  # noqa: BLE001
            pass
        raise LLMError(f"HTTP {err.code} from {url}: {detail or err.reason}") from err
    except urllib.error.URLError as err:
        raise LLMError(f"cannot reach {url}: {err.reason}") from err


def _get(url: str, headers: dict) -> dict:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            return json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raise LLMError(f"HTTP {err.code} listing models: {err.reason}") from err
    except urllib.error.URLError as err:
        raise LLMError(f"cannot reach {url}: {err.reason}") from err


def base_url_for(provider: str, custom_base_url: str = "") -> str:
    override = (custom_base_url or "").strip().rstrip("/")
    if override:
        return override
    url = get_provider(provider).base_url.rstrip("/")
    if not url:
        raise LLMError("This provider needs a custom_base_url.")
    return url


def _headers(prov: Provider, key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    headers.update(prov.extra_headers or {})
    if prov.dialect == "anthropic":
        headers["x-api-key"] = key
        headers["anthropic-version"] = "2023-06-01"
    elif prov.dialect == "google":
        pass  # key goes on the query string
    elif key:
        headers["Authorization"] = f"Bearer {key}"
    if prov.key == "openrouter":
        headers.setdefault("HTTP-Referer", "https://github.com/comfyanonymous/ComfyUI")
        headers.setdefault("X-Title", "ComfyUI Alchemist")
    return headers


def list_models(provider: str, inline_key: str = "", custom_base_url: str = "") -> list[str]:
    prov = get_provider(provider)
    key = resolve_key(provider, inline_key)
    if prov.needs_key and not key:
        return list(prov.fallback_models)

    try:
        base = base_url_for(provider, custom_base_url)
    except LLMError:
        return list(prov.fallback_models)

    try:
        if prov.dialect == "google":
            data = _get(f"{base}/models?key={key}", {"Content-Type": "application/json"})
            names = [
                str(m.get("name", "")).split("/")[-1]
                for m in data.get("models", [])
                if "generateContent" in (m.get("supportedGenerationMethods") or [])
            ]
        elif prov.aliases_path:
            names = _named_with_aliases(base, prov, key)
        else:
            data = _get(f"{base}{prov.models_path}", _headers(prov, key))
            rows = data.get("data") or data.get("models") or []
            names = [str(r.get("id") or r.get("name") or "") for r in rows]
        # Image/video/embedding endpoints share the listing and 400 on chat.
        names = sorted({n for n in names if n and is_chat_model(n)})
        return names or list(prov.fallback_models)
    except LLMError:
        return list(prov.fallback_models)


def _named_with_aliases(base: str, prov: Provider, key: str) -> list[str]:
    """Model names from the alias endpoint, preferring the readable alias.

    /v1/models only returns dated snapshot ids. The alias endpoint also lists the
    short names people actually type, so surface those in the dropdown.
    """
    try:
        data = _get(f"{base}{prov.aliases_path}", _headers(prov, key))
    except LLMError:
        data = _get(f"{base}{prov.models_path}", _headers(prov, key))
        rows = data.get("data") or data.get("models") or []
        return [str(r.get("id") or r.get("name") or "") for r in rows]

    names: list[str] = []
    for row in data.get("models") or data.get("data") or []:
        if not isinstance(row, dict):
            continue
        canonical = str(row.get("id") or row.get("name") or "")
        if canonical:
            names.append(prettiest_name(canonical, row.get("aliases") or []))
    return names


def image_to_data_url(tensor) -> str | None:
    """First frame of a ComfyUI IMAGE tensor as a JPEG data URL."""
    if tensor is None:
        return None
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None

    try:
        arr = tensor
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            return None
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        img = Image.fromarray(arr[:, :, :3], "RGB")
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def file_to_data_url(path: str) -> str | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def _split_data_url(data_url: str) -> tuple[str, str]:
    head, _, b64 = data_url.partition(",")
    media = head.split(":")[-1].split(";")[0] or "image/jpeg"
    return media, b64


def _openai_payload(model, system, user, images, temperature, max_tokens, extra):
    content: list[dict] = [{"type": "text", "text": user}]
    for url in images:
        content.append({"type": "image_url", "image_url": {"url": url}})
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content if images else user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload.update(extra)
    return payload


def _anthropic_payload(model, system, user, images, temperature, max_tokens, extra):
    content: list[dict] = []
    for url in images:
        media, b64 = _split_data_url(url)
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media, "data": b64},
            }
        )
    content.append({"type": "text", "text": user})
    payload = {
        "model": model,
        "system": system,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload.update(extra)
    return payload


def _google_payload(model, system, user, images, temperature, max_tokens, extra):
    del model
    parts: list[dict] = [{"text": user}]
    for url in images:
        media, b64 = _split_data_url(url)
        parts.append({"inline_data": {"mime_type": media, "data": b64}})
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        "safetySettings": [
            {"category": c, "threshold": "BLOCK_NONE"}
            for c in (
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            )
        ],
    }
    payload.update(extra)
    return payload


def _extract(dialect: str, data: dict) -> tuple[str, str]:
    """Return (text, reasoning)."""
    if dialect == "anthropic":
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        think = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
        return text.strip(), think.strip()

    if dialect == "google":
        cands = data.get("candidates") or []
        if not cands:
            blocked = (data.get("promptFeedback") or {}).get("blockReason")
            if blocked:
                raise LLMError(f"Gemini blocked this prompt: {blocked}")
            return "", ""
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        if not text.strip() and cands[0].get("finishReason") == "SAFETY":
            raise LLMError("Gemini blocked the response (SAFETY). Try xAI or a local model.")
        return text.strip(), ""

    choices = data.get("choices") or []
    if not choices:
        raise LLMError(f"No choices in response: {json.dumps(data)[:300]}")
    msg = choices[0].get("message") or {}
    text = msg.get("content") or ""
    if isinstance(text, list):
        text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
    think = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return str(text).strip(), str(think).strip()


def _rejects_reasoning(message: str) -> bool:
    low = message.lower()
    return "reasoning" in low and ("support" in low or "invalid" in low or "unknown" in low)


def chat(
    provider: str,
    model: str,
    system: str,
    user: str,
    images: list[str] | None = None,
    temperature: float = 0.8,
    max_tokens: int = 1400,
    inline_key: str = "",
    custom_base_url: str = "",
    extra: dict | None = None,
    reasoning: str = "provider default",
) -> tuple[str, str, str]:
    """Returns (text, reasoning, model_used)."""
    prov = get_provider(provider)
    key = resolve_key(provider, inline_key)
    if prov.needs_key and not key:
        env_hint = " / ".join(prov.env_vars) or "the node"
        raise LLMError(
            f"No API key for {prov.label}. Paste one and hit VAULT KEY, or set {env_hint}."
        )

    images = [u for u in (images or []) if u]
    if images and not prov.vision:
        raise LLMError(f"{prov.label} has no vision. Switch provider or unplug the image.")

    base = base_url_for(provider, custom_base_url)
    model = (model or "").strip() or (prov.fallback_models[0] if prov.fallback_models else "")
    if not model:
        raise LLMError("Pick a model first.")

    options = dict(extra or {})
    effort = (reasoning or "").strip().lower()
    if prov.reasoning_param and effort and effort != "provider default":
        options.setdefault("reasoning_effort", effort)

    builders = {
        "openai": _openai_payload,
        "anthropic": _anthropic_payload,
        "google": _google_payload,
    }

    if prov.dialect == "anthropic":
        url = f"{base}/messages"
    elif prov.dialect == "google":
        url = f"{base}/models/{model}:generateContent?key={key}"
    else:
        url = f"{base}/chat/completions"

    def send(opts: dict) -> dict:
        payload = builders[prov.dialect](
            model, system, user, images, float(temperature), int(max_tokens), opts
        )
        return _post(url, payload, _headers(prov, key))

    try:
        data = send(options)
    except LLMError as err:
        # Not every model on a reasoning-capable provider accepts the parameter.
        if "reasoning_effort" in options and _rejects_reasoning(str(err)):
            options.pop("reasoning_effort")
            data = send(options)
        else:
            raise

    text, reasoning_text = _extract(prov.dialect, data)
    if not text:
        raise LLMError("The model returned nothing. Try another model or lower max_tokens.")
    return text, reasoning_text, model
