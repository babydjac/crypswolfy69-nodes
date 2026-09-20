"""Live Hugging Face browsing, plus an aria2 install helper.

The curated index in `server.py` is a slow, filtered, cached view of a handful of
trusted owners. This module is the opposite: it queries the Hugging Face API on every
keystroke so you can reach any repo on the Hub, then lists that repo's real file tree
with sizes and a ComfyUI destination folder for each file.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.parse import quote, urlencode

import aiohttp
from aiohttp import web
from server import PromptServer

from . import server as core

SEARCH_CACHE_TTL = 60
TREE_CACHE_TTL = 300
_SEARCH_CACHE: Dict[str, tuple] = {}
_TREE_CACHE: Dict[str, tuple] = {}
_CACHE_MAX_ENTRIES = 256

SORT_FIELDS = {
    "trending": "trendingScore",
    "downloads": "downloads",
    "likes": "likes",
    "modified": "lastModified",
    "created": "createdAt",
}

PIPELINE_FILTERS = {
    "any": None,
    "text-to-image": "text-to-image",
    "image-to-image": "image-to-image",
    "text-to-video": "text-to-video",
    "image-to-video": "image-to-video",
    "text-generation": "text-generation",
}


def _trim(cache: Dict[str, tuple]) -> None:
    if len(cache) <= _CACHE_MAX_ENTRIES:
        return
    for key in sorted(cache, key=lambda k: cache[k][0])[: len(cache) // 2]:
        cache.pop(key, None)


def _cached(cache: Dict[str, tuple], key: str, ttl: int):
    hit = cache.get(key)
    if hit and (time.time() - hit[0]) < ttl:
        return hit[1]
    return None


def _store(cache: Dict[str, tuple], key: str, value) -> None:
    cache[key] = (time.time(), value)
    _trim(cache)


def _repo_row(repo: Dict[str, object]) -> Dict[str, object]:
    repo_id = str(repo.get("id") or repo.get("modelId") or "")
    owner, _, name = repo_id.partition("/")
    tags = [str(t) for t in (repo.get("tags") or [])]
    return {
        "repo_id": repo_id,
        "owner": owner or "",
        "name": name or repo_id,
        "downloads": int(repo.get("downloads") or 0),
        "likes": int(repo.get("likes") or 0),
        "trending": float(repo.get("trendingScore") or 0),
        "modified": str(repo.get("lastModified") or ""),
        "pipeline": str(repo.get("pipeline_tag") or ""),
        "gated": bool(repo.get("gated")),
        "private": bool(repo.get("private")),
        "tags": tags[:12],
        "url": f"{core.HF_WEB}/{repo_id}",
    }


async def search_models(
    query: str,
    sort: str = "trending",
    limit: int = 40,
    author: str = "",
    pipeline: str = "any",
) -> List[Dict[str, object]]:
    params = {
        "limit": max(1, min(int(limit or 40), 100)),
        "sort": SORT_FIELDS.get(sort, "trendingScore"),
        "direction": -1,
        "full": "true",
    }
    query = (query or "").strip()
    if query:
        params["search"] = query
    author = (author or "").strip()
    if author:
        params["author"] = author
    tag = PIPELINE_FILTERS.get(pipeline or "any")
    if tag:
        params["pipeline_tag"] = tag

    url = f"{core.HF_API}/models?{urlencode(params)}"
    cache_key = url
    hit = _cached(_SEARCH_CACHE, cache_key, SEARCH_CACHE_TTL)
    if hit is not None:
        return hit

    # request_json applies the HF token itself and falls back to anonymous on 401.
    timeout = aiohttp.ClientTimeout(total=25)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = await core.request_json(session, url, timeout=25)

    rows = [_repo_row(r) for r in (payload or []) if isinstance(r, dict)]
    rows = [r for r in rows if r["repo_id"]]
    _store(_SEARCH_CACHE, cache_key, rows)
    return rows


def _model_file(repo: Dict[str, object], entry: Dict[str, object]) -> Optional[Dict[str, object]]:
    path = str(entry.get("path") or "")
    if not path:
        return None
    suffix = Path(path).suffix.lower()
    if suffix not in core.MODEL_EXTENSIONS:
        return None

    size = entry.get("size")
    if size is None:
        lfs = entry.get("lfs")
        if isinstance(lfs, dict):
            size = lfs.get("size")
    size = int(size or 0)

    repo_id = str(repo.get("id") or "")
    category = core.categorize_file(repo, path)
    return {
        "id": f"live::{repo_id}::{path}",
        "repo_id": repo_id,
        "repo_name": repo_id.split("/")[-1],
        "repo_revision": core._repo_tree_revision(repo),
        "path": path,
        "filename": Path(path).name,
        "size": size,
        "category": category,
        "title": core.title_from_path(path, repo_id),
        "family": core.detect_family(repo_id, path),
        "shard": core.is_shard_file(path),
        "url": f"{core.HF_WEB}/{repo_id}/resolve/main/{quote(path)}",
    }


def _installed_in_category_root(row: Dict[str, object]) -> Optional[str]:
    """Match weights sitting directly in models/<category>/ (or one level under it).

    `installed_path_for_item` only looks in models/<category>/<family>/, which is where
    this extension installs things. Files fetched by hand or by a script usually land in
    the category root instead, and reporting those as missing invites a pointless re-download.
    """
    filename = str(row.get("filename") or "")
    if not filename:
        return None
    category = str(row.get("category") or "checkpoints")
    root = core.models_root() / category
    if not root.is_dir():
        return None

    size = row.get("size")
    expected = size if isinstance(size, int) and size > 0 else None

    def match(path: Path) -> Optional[str]:
        try:
            if expected is not None and path.stat().st_size != expected:
                return None
        except OSError:
            return None
        return str(path)

    direct = root / filename
    if direct.is_file():
        hit = match(direct)
        if hit:
            return hit

    try:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            nested = child / filename
            if nested.is_file():
                hit = match(nested)
                if hit:
                    return hit
    except OSError:
        return None
    return None


async def repo_files(repo_id: str) -> Dict[str, object]:
    repo_id = (repo_id or "").strip().strip("/")
    if not repo_id or repo_id.count("/") != 1:
        raise ValueError("Need a repo id shaped like owner/name.")

    hit = _cached(_TREE_CACHE, repo_id, TREE_CACHE_TTL)
    if hit is not None:
        return hit

    timeout = aiohttp.ClientTimeout(total=40)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        info = await core.request_json(
            session, f"{core.HF_API}/models/{quote(repo_id, safe='/')}", timeout=30
        )
        if not isinstance(info, dict):
            raise ValueError(f"Repo not found: {repo_id}")
        revision = core._repo_tree_revision(info)
        tree = await core.fetch_repo_tree(session, repo_id, revision)

    files = [f for f in (_model_file(info, e) for e in (tree or [])) if f]
    files.sort(key=lambda f: (-f["size"], f["path"]))

    installed = core.build_installed_lookup(files)
    for row in files:
        found = core.installed_path_for_item(row, installed)
        if not found:
            found = _installed_in_category_root(row)
        row["installed_path"] = found or None
        row["installed"] = bool(found)
        row["destination"] = core.preview_destination(row)

    by_category: Dict[str, int] = {}
    for row in files:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1

    result = {
        "repo": _repo_row(info),
        "revision": revision,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(f["size"] for f in files),
        "categories": sorted(by_category.items(), key=lambda kv: (-kv[1], kv[0])),
        "readme_url": f"{core.HF_WEB}/{repo_id}",
    }
    _store(_TREE_CACHE, repo_id, result)
    return result


def aria2_state() -> Dict[str, object]:
    path = shutil.which("aria2c")
    version = ""
    if path:
        try:
            out = subprocess.run(
                [path, "--version"], capture_output=True, text=True, timeout=10, check=False
            )
            version = (out.stdout or "").splitlines()[0].strip()
        except Exception:  # noqa: BLE001
            version = "installed"
    return {"installed": bool(path), "path": path or "", "version": version}


def install_aria2() -> Dict[str, object]:
    """apt-get install aria2, skipping third-party lists that often break update."""
    state = aria2_state()
    if state["installed"]:
        return {"ok": True, "already": True, **state}

    if not shutil.which("apt-get"):
        return {"ok": False, "error": "apt-get not available on this image."}

    env = {**core.os.environ, "DEBIAN_FRONTEND": "noninteractive"}
    logs: List[str] = []
    update = subprocess.run(
        [
            "apt-get", "update",
            "-o", "Dir::Etc::sourcelist=sources.list",
            "-o", "Dir::Etc::sourceparts=-",
            "-o", "APT::Get::List-Cleanup=0",
        ],
        capture_output=True, text=True, timeout=300, check=False, env=env,
    )
    logs.append(f"apt-get update -> {update.returncode}")
    install = subprocess.run(
        ["apt-get", "install", "-y", "aria2"],
        capture_output=True, text=True, timeout=600, check=False, env=env,
    )
    logs.append(f"apt-get install aria2 -> {install.returncode}")

    state = aria2_state()
    if state["installed"]:
        return {"ok": True, "already": False, "logs": logs, **state}
    tail = (install.stderr or install.stdout or "").strip().splitlines()[-6:]
    return {"ok": False, "error": "install finished but aria2c is still missing", "logs": logs + tail}


def register_routes() -> None:
    routes = PromptServer.instance.routes

    @routes.get("/hf-model-downloader/search")
    async def hf_live_search(request: web.Request) -> web.Response:
        try:
            rows = await search_models(
                query=request.query.get("q", ""),
                sort=request.query.get("sort", "trending"),
                limit=core.parse_int(request.query.get("limit"), 40, 1, 100),
                author=request.query.get("author", ""),
                pipeline=request.query.get("pipeline", "any"),
            )
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(exc)}, status=200)
        return web.json_response(
            {"ok": True, "count": len(rows), "results": rows, "sorts": sorted(SORT_FIELDS)}
        )

    @routes.get("/hf-model-downloader/repo")
    async def hf_live_repo(request: web.Request) -> web.Response:
        try:
            data = await repo_files(request.query.get("id", ""))
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=200)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(exc)}, status=200)
        return web.json_response({"ok": True, **data})

    @routes.get("/hf-model-downloader/aria2")
    async def hf_aria2_status(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True, **aria2_state()})

    @routes.post("/hf-model-downloader/aria2")
    async def hf_aria2_install(_request: web.Request) -> web.Response:
        result = await asyncio.get_running_loop().run_in_executor(None, install_aria2)
        return web.json_response(result)


register_routes()
