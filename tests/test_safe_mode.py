"""Tests for the watchdog safe-mode contract on the eval component.

Per specs/openapi/eval.yaml: when started with
`EUGENE_PLEXUS_EVAL_SAFE_MODE=1` the eval component must

  - skip loading its persisted config file (defaults only)
  - still expose /v1/config endpoints
  - report /healthz as `degraded` with `safeMode: true`
  - allow PATCH /v1/config to write to the on-disk file as normal
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_eval.app import create_app
from eugene_plexus_eval.settings import Settings


@pytest.fixture
def safe_mode_settings(tmp_path: Path) -> Settings:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"minLogitEntropy": 3.0, "logLevel": "DEBUG"}),
        encoding="utf-8",
    )
    return Settings(config_file=config, safe_mode=True)


@pytest.fixture
def safe_mode_app(safe_mode_settings: Settings) -> FastAPI:
    return create_app(settings=safe_mode_settings)


@pytest.fixture
def safe_mode_client(safe_mode_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(safe_mode_app) as c:
        yield c


def test_healthz_reports_safe_mode_and_degraded(safe_mode_client: TestClient) -> None:
    response = safe_mode_client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["safeMode"] is True


def test_config_get_returns_defaults_not_disk_values(safe_mode_client: TestClient) -> None:
    response = safe_mode_client.get("/v1/config")
    assert response.status_code == 200
    body = response.json()
    # Disk had minLogitEntropy=3.0 and logLevel=DEBUG; safe mode must serve
    # the built-in defaults instead.
    assert body.get("logLevel") == "INFO"
    assert body.get("minLogitEntropy") != 3.0


def test_patch_config_writes_to_disk_in_safe_mode(
    safe_mode_client: TestClient, safe_mode_settings: Settings
) -> None:
    response = safe_mode_client.patch("/v1/config", json={"logLevel": "WARNING"})
    assert response.status_code == 200
    body = response.json()
    assert "logLevel" in body["applied"]

    on_disk = yaml.safe_load(safe_mode_settings.config_file.read_text(encoding="utf-8"))
    assert on_disk["logLevel"] == "WARNING"


def test_executing_endpoints_return_503_in_safe_mode(safe_mode_client: TestClient) -> None:
    from uuid import uuid4

    run = safe_mode_client.post(
        "/v1/eval/runs", json={"evalSuiteId": str(uuid4()), "checkpointId": str(uuid4())}
    )
    assert run.status_code == 503
    assert run.json()["component"] == "eval"
    assert run.headers["content-type"].startswith("application/problem+json")

    compare = safe_mode_client.post(
        "/v1/eval/compare",
        json={
            "evalSuiteId": str(uuid4()),
            "baselineCheckpointId": str(uuid4()),
            "candidateCheckpointId": str(uuid4()),
        },
    )
    assert compare.status_code == 503

    create = safe_mode_client.post(
        "/v1/eval/suites",
        json={"evalSuiteId": str(uuid4()), "name": "x", "metrics": ["perplexity"]},
    )
    assert create.status_code == 503


def test_suite_list_degrades_to_empty_in_safe_mode(safe_mode_client: TestClient) -> None:
    # The read endpoint stays available (returns an empty list) even with the
    # engine unbuilt, so callers polling for suites get a valid result.
    response = safe_mode_client.get("/v1/eval/suites")
    assert response.status_code == 200
    assert response.json() == {"suites": []}
