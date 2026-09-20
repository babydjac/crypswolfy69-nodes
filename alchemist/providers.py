"""LLM provider registry: base URLs, auth, model listing and request shape.

Three request dialects are supported:
  openai     — /chat/completions, works for xAI, OpenAI, OpenRouter, DeepSeek,
               Groq, Mistral, Together, Ollama and any OpenAI-compatible endpoint
  anthropic  — /v1/messages
  google     — /v1beta/models/<model>:generateContent
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Sequence

VAULT_DIR = "/workspace/.config/llm_vault"


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    dialect: str
    base_url: str
    env_vars: tuple[str, ...] = ()
    models_path: str = "/models"
    fallback_models: tuple[str, ...] = ()
    vision: bool = True
    needs_key: bool = True
    notes: str = ""
    extra_headers: dict = field(default_factory=dict)
    # Accepts an OpenAI-style reasoning_effort. Reasoning models are 4-5x slower
    # without it, so the node sends it whenever the provider understands it.
    reasoning_param: bool = False
    # Endpoint listing alias names alongside canonical ids. /v1/models returns
    # dated snapshots like grok-4.20-0309-non-reasoning; the alias list has the
    # readable grok-4.20-non-reasoning that people actually type.
    aliases_path: str = ""


# Substrings identifying models that cannot serve a chat completion. Image and
# video endpoints appear in the same /models listing and return HTTP 400.
NON_CHAT_MODEL_HINTS = (
    "imagine-image",
    "imagine-video",
    "multi-agent",
    "-tts",
    "whisper",
    "embed",
    "moderation",
    "dall-e",
    "stable-diffusion",
    "rerank",
)


def is_chat_model(name: str) -> bool:
    low = (name or "").strip().lower()
    if not low:
        return False
    return not any(hint in low for hint in NON_CHAT_MODEL_HINTS)


# A bare 4-digit run is a date stamp (0309, 0825), not a version number. Versions
# always carry a dot, so requiring no adjacent digit or dot keeps "4.20" intact.
_DATE_STAMP = re.compile(r"(?<![\d.])\d{4,8}(?![\d.])")

# Alias noise that adds length without telling you anything useful.
_ALIAS_NOISE = ("latest", "beta", "experimental", "-gv", "preview", "snapshot")


def prettiest_name(canonical: str, aliases: Sequence[str] | None = None) -> str:
    """Pick the most readable name that still resolves for a model.

    Prefers a short, undated, noise-free alias over a dated canonical id, so the
    dropdown shows `grok-4.20-non-reasoning` rather than
    `grok-4.20-0309-non-reasoning`. Falls back to the canonical id.
    """
    canonical = (canonical or "").strip()
    if not canonical:
        return ""

    def clean(name: str) -> bool:
        low = name.lower()
        if _DATE_STAMP.search(low):
            return False
        return not any(noise in low for noise in _ALIAS_NOISE)

    candidates = [a.strip() for a in (aliases or []) if a and a.strip()]
    # An alias must describe the same variant, so reasoning and non-reasoning
    # models never collapse onto each other's names.
    wants_non_reasoning = "non-reasoning" in canonical.lower()
    candidates = [
        a for a in candidates if ("non-reasoning" in a.lower()) == wants_non_reasoning
    ]
    usable = sorted((a for a in candidates if clean(a)), key=lambda a: (len(a), a))
    if usable:
        return usable[0]
    return canonical


PROVIDERS: dict[str, Provider] = {
    "xai": Provider(
        key="xai",
        label="xAI / Grok",
        dialect="openai",
        base_url="https://api.x.ai/v1",
        env_vars=("XAI_API_KEY", "GROK_API_KEY"),
        fallback_models=(
            "grok-4.20-non-reasoning",
            "grok-4.5",
            "grok-4.20",
            "grok-4.6",
            "grok-4.3",
        ),
        reasoning_param=True,
        aliases_path="/language-models",
        notes=(
            "Loosest content policy. Measured on the caption task: "
            "grok-4.20-non-reasoning 2.1s, grok-4.5 5.8s, grok-4.6 12s, grok-4.20 17s."
        ),
    ),
    "openai": Provider(
        key="openai",
        label="OpenAI",
        dialect="openai",
        base_url="https://api.openai.com/v1",
        env_vars=("OPENAI_API_KEY",),
        fallback_models=("gpt-5.2", "gpt-5.2-mini", "gpt-4.1", "gpt-4o"),
        reasoning_param=True,
        notes="Refuses explicit adult content.",
    ),
    "anthropic": Provider(
        key="anthropic",
        label="Anthropic / Claude",
        dialect="anthropic",
        base_url="https://api.anthropic.com/v1",
        env_vars=("ANTHROPIC_API_KEY",),
        models_path="/models",
        fallback_models=(
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-1-20250805",
            "claude-3-5-haiku-20241022",
        ),
        notes="Refuses explicit adult content.",
    ),
    "google": Provider(
        key="google",
        label="Google / Gemini",
        dialect="google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        fallback_models=("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"),
        notes="Safety filters block most explicit content.",
    ),
    "openrouter": Provider(
        key="openrouter",
        label="OpenRouter (any model)",
        dialect="openai",
        base_url="https://openrouter.ai/api/v1",
        env_vars=("OPENROUTER_API_KEY",),
        reasoning_param=True,
        fallback_models=(
            "x-ai/grok-4.6",
            "anthropic/claude-sonnet-4.5",
            "google/gemini-2.5-pro",
            "meta-llama/llama-4-maverick",
            "qwen/qwen3-vl-235b-a22b-instruct",
        ),
        notes="One key, every model. Uncensored community models live here.",
    ),
    "deepseek": Provider(
        key="deepseek",
        label="DeepSeek",
        dialect="openai",
        base_url="https://api.deepseek.com/v1",
        env_vars=("DEEPSEEK_API_KEY",),
        fallback_models=("deepseek-chat", "deepseek-reasoner"),
        vision=False,
        notes="Text only — no vision captioning.",
    ),
    "groq": Provider(
        key="groq",
        label="Groq (fast)",
        dialect="openai",
        base_url="https://api.groq.com/openai/v1",
        env_vars=("GROQ_API_KEY",),
        fallback_models=(
            "llama-4-maverick-17b-128e-instruct",
            "llama-3.3-70b-versatile",
        ),
        notes="Very fast. Good for bulk dataset captioning.",
    ),
    "mistral": Provider(
        key="mistral",
        label="Mistral",
        dialect="openai",
        base_url="https://api.mistral.ai/v1",
        env_vars=("MISTRAL_API_KEY",),
        fallback_models=("mistral-large-latest", "pixtral-large-latest"),
    ),
    "together": Provider(
        key="together",
        label="Together AI",
        dialect="openai",
        base_url="https://api.together.xyz/v1",
        env_vars=("TOGETHER_API_KEY",),
        fallback_models=(
            "Qwen/Qwen3-VL-235B-A22B-Instruct",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        ),
        notes="Hosts open VLMs that will caption adult content.",
    ),
    "ollama": Provider(
        key="ollama",
        label="Ollama (local, free)",
        dialect="openai",
        base_url="http://127.0.0.1:11434/v1",
        env_vars=(),
        needs_key=False,
        fallback_models=("qwen3-vl:8b", "llama3.2-vision:11b", "llava:13b"),
        notes="Runs on this box. No key, no filter, no API bill.",
    ),
    "vllm": Provider(
        key="vllm",
        label="Local vLLM / LM Studio",
        dialect="openai",
        base_url="http://127.0.0.1:8000/v1",
        env_vars=(),
        needs_key=False,
        fallback_models=("local-model",),
        notes="Any OpenAI-compatible server you host. Override base_url.",
    ),
    "custom": Provider(
        key="custom",
        label="Custom OpenAI-compatible",
        dialect="openai",
        base_url="",
        env_vars=("CUSTOM_LLM_API_KEY",),
        needs_key=False,
        fallback_models=(),
        notes="Set custom_base_url on the node.",
    ),
}

PROVIDER_ORDER = list(PROVIDERS.keys())


def get_provider(name: str) -> Provider:
    key = (name or "xai").strip().lower()
    if key in PROVIDERS:
        return PROVIDERS[key]
    for prov in PROVIDERS.values():
        if prov.label.lower() == key:
            return prov
    return PROVIDERS["xai"]


def vault_path(provider: str) -> str:
    return os.path.join(VAULT_DIR, f"{get_provider(provider).key}.key")


def save_key(provider: str, api_key: str) -> bool:
    key = (api_key or "").strip()
    if not key:
        return False
    os.makedirs(VAULT_DIR, mode=0o700, exist_ok=True)
    path = vault_path(provider)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(key)
    os.chmod(path, 0o600)
    return True


def resolve_key(provider: str, inline: str = "") -> str:
    """Inline key wins, then the on-disk vault, then environment variables."""
    direct = (inline or "").strip()
    if direct:
        return direct

    path = vault_path(provider)
    try:
        with open(path, encoding="utf-8") as handle:
            stored = handle.read().strip()
        if stored:
            return stored
    except OSError:
        pass

    for env_var in get_provider(provider).env_vars:
        val = (os.environ.get(env_var) or "").strip()
        if val:
            return val
    return ""


def key_status(provider: str) -> dict:
    prov = get_provider(provider)
    path = vault_path(provider)
    vaulted = False
    try:
        with open(path, encoding="utf-8") as handle:
            vaulted = bool(handle.read().strip())
    except OSError:
        vaulted = False
    env_hit = next((v for v in prov.env_vars if (os.environ.get(v) or "").strip()), "")
    return {
        "provider": prov.key,
        "label": prov.label,
        "vaulted": vaulted,
        "env_var": env_hit,
        "needs_key": prov.needs_key,
        "ready": bool(vaulted or env_hit or not prov.needs_key),
        "vision": prov.vision,
        "notes": prov.notes,
    }
