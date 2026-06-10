"""Config protocol routes: GET, PATCH, schema, test."""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import APIRouter, Request

from .._generated.common_models import (
    ConfigDocument,
    ConfigSchema,
    ConfigTestRequest,
    ConfigTestResult,
    ConfigUpdateRequest,
    ConfigUpdateResult,
)
from ..config import ConfigStore, as_schema

router = APIRouter(tags=["config"])


@router.get("/v1/config", response_model=ConfigDocument)
async def get_config(request: Request) -> ConfigDocument:
    store: ConfigStore = request.app.state.config_store
    return store.as_document()


@router.get("/v1/config/schema", response_model=ConfigSchema)
async def get_config_schema() -> ConfigSchema:
    return as_schema()


@router.patch("/v1/config", response_model=ConfigUpdateResult)
async def patch_config(
    request: Request,
    body: ConfigUpdateRequest,
) -> ConfigUpdateResult:
    store: ConfigStore = request.app.state.config_store
    return store.apply_patch(body)


def _dir_writable(path_str: str | None) -> str | None:
    """Return None if the directory is writable, else a human error string."""
    if not path_str:
        return "(unset)"
    path = Path(path_str)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return f"cannot create {path}: {e}"
    if not os.access(path, os.W_OK):
        return f"{path} is not writable"
    return None


@router.post("/v1/config/test", response_model=ConfigTestResult)
async def test_config(
    request: Request,
    body: ConfigTestRequest | None = None,
) -> ConfigTestResult:
    """Verify the eval output root is writable (and, when the eval engine
    lands, that referenced checkpoints / datasets are reachable).

    The v0.3 skeleton has no engine to probe, so this checks the eval
    output root the engine will write to — the cheap, real part of the
    contract — and reports success/failure in the standard
    `ConfigTestResult` shape. Future engine work adds the
    checkpoint/dataset-reachability probe here.
    """
    start = time.perf_counter()
    # Body overrides are accepted for protocol uniformity but the skeleton
    # tests the saved config as-is; the engine work will honor overrides.
    _ = body
    store: ConfigStore = request.app.state.config_store

    problems: list[str] = []
    err = _dir_writable(store.get("evalRoot"))
    if err is not None:
        problems.append(f"eval output root: {err}")

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    if problems:
        return ConfigTestResult(
            ok=False,
            component="eval",
            latencyMs=elapsed_ms,
            error="; ".join(problems),
        )
    return ConfigTestResult(
        ok=True,
        component="eval",
        latencyMs=elapsed_ms,
        summary=(
            "Eval output root is writable. Checkpoint and dataset "
            "reachability is checked once the eval engine is implemented."
        ),
    )
