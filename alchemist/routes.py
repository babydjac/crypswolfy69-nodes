"""HTTP routes for the Alchemist node chrome."""

from __future__ import annotations

try:
    from .llm_client import LLMError, chat, list_models
    from .prompts import build_system_prompt, build_user_message
    from .providers import PROVIDER_ORDER, PROVIDERS, key_status, save_key
except ImportError:
    from llm_client import LLMError, chat, list_models
    from prompts import build_system_prompt, build_user_message
    from providers import PROVIDER_ORDER, PROVIDERS, key_status, save_key

_REGISTERED = False


def register_routes() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return

    routes = getattr(PromptServer.instance, "routes", None)
    if routes is None:
        return

    _REGISTERED = True

    @routes.get("/alchemist/providers")
    async def providers(_request):
        payload = []
        for key in PROVIDER_ORDER:
            prov = PROVIDERS[key]
            row = {
                "key": key,
                "label": prov.label,
                "dialect": prov.dialect,
                "base_url": prov.base_url,
                "fallback_models": list(prov.fallback_models),
            }
            row.update(key_status(key))
            payload.append(row)
        return web.json_response(payload)

    @routes.get("/alchemist/models")
    async def models(request):
        provider = request.query.get("provider", "xai")
        inline = request.query.get("key", "")
        base = request.query.get("base_url", "")
        try:
            return web.json_response(
                {"ok": True, "models": list_models(provider, inline, base)}
            )
        except Exception as err:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(err)})

    @routes.get("/alchemist/key_status")
    async def key_state(request):
        provider = request.query.get("provider", "")
        if provider:
            return web.json_response(key_status(provider))
        return web.json_response({k: key_status(k) for k in PROVIDER_ORDER})

    @routes.post("/alchemist/vault")
    async def vault(request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        provider = str(body.get("provider") or "xai")
        api_key = str(body.get("api_key") or "")
        if not api_key.strip():
            return web.json_response({"ok": False, "error": "empty key"})
        save_key(provider, api_key)
        models_found = list_models(provider, api_key, str(body.get("base_url") or ""))
        return web.json_response(
            {
                "ok": True,
                "provider": provider,
                "count": len(models_found),
                "models": models_found,
                "status": key_status(provider),
            }
        )

    @routes.post("/alchemist/preview")
    async def preview(request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}

        provider = str(body.get("provider") or "xai")
        engine = str(body.get("target_engine") or "krea2")
        mode = str(body.get("caption_mode") or "regular")
        system = build_system_prompt(
            engine=engine,
            caption_mode=mode,
            nsfw=bool(body.get("nsfw", True)),
            controlnet_lock=str(body.get("controlnet_lock") or "auto"),
            has_control_image=False,
            lora_kind=str(body.get("lora_kind") or "character"),
            trigger_word=str(body.get("trigger_word") or ""),
            caption_target_words=int(body.get("caption_words") or 45),
        )
        user = build_user_message(
            seed=str(body.get("simple_prompt") or ""),
            extra=str(body.get("extra_direction") or ""),
            engine=engine,
            caption_mode=mode,
            edit_instruction=str(body.get("edit_instruction") or ""),
            clip_seconds=float(body.get("clip_seconds") or 8.0),
            aspect=str(body.get("h3_aspect") or "16:9"),
            music=str(body.get("h3_music") or "auto"),
            trigger_word=str(body.get("trigger_word") or ""),
            lora_kind=str(body.get("lora_kind") or "character"),
            omit_extra=str(body.get("omit_terms") or ""),
        )
        try:
            text, reasoning, used = chat(
                provider=provider,
                model=str(body.get("model") or ""),
                system=system,
                user=user,
                images=[],
                temperature=float(body.get("temperature") or 0.8),
                max_tokens=int(body.get("max_tokens") or 1400),
                inline_key=str(body.get("api_key") or ""),
                custom_base_url=str(body.get("custom_base_url") or ""),
                reasoning=str(body.get("reasoning") or "low"),
            )
        except LLMError as err:
            return web.json_response({"ok": False, "error": str(err)})
        except Exception as err:  # noqa: BLE001
            return web.json_response({"ok": False, "error": f"{type(err).__name__}: {err}"})

        return web.json_response(
            {"ok": True, "prompt": text, "reasoning": reasoning, "model": used}
        )
