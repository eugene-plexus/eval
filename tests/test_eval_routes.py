"""Tests for the eval domain routes (v0.3 skeleton).

The engine-dependent endpoints (start a run, read a result, compare
checkpoints) return 501 (engine not implemented). The suite endpoints are
the engine-independent wire shape: list returns empty, create echoes the
posted suite, get/delete 404 (nothing persisted yet).
"""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def test_list_eval_suites_returns_empty(client: TestClient) -> None:
    response = client.get("/v1/eval/suites")
    assert response.status_code == 200
    assert response.json() == {"suites": []}


def test_create_eval_suite_echoes_wire_shape(client: TestClient) -> None:
    suite_id = str(uuid4())
    body = {
        "evalSuiteId": suite_id,
        "name": "neutral-smoke",
        "prompts": ["The capital of France is", "Two plus two equals"],
        "metrics": ["perplexity", "token_entropy"],
    }
    response = client.post("/v1/eval/suites", json=body)
    assert response.status_code == 201
    echoed = response.json()
    assert echoed["evalSuiteId"] == suite_id
    assert echoed["name"] == "neutral-smoke"
    assert echoed["metrics"] == ["perplexity", "token_entropy"]


def test_create_eval_suite_rejects_bad_metric(client: TestClient) -> None:
    body = {
        "evalSuiteId": str(uuid4()),
        "name": "bad-metric",
        "metrics": ["not_a_real_metric"],
    }
    response = client.post("/v1/eval/suites", json=body)
    assert response.status_code == 422


def test_get_eval_suite_returns_404(client: TestClient) -> None:
    response = client.get(f"/v1/eval/suites/{uuid4()}")
    assert response.status_code == 404
    body = response.json()
    assert body["component"] == "eval"


def test_delete_eval_suite_returns_404(client: TestClient) -> None:
    response = client.delete(f"/v1/eval/suites/{uuid4()}")
    assert response.status_code == 404


def test_start_eval_run_returns_501(client: TestClient) -> None:
    response = client.post(
        "/v1/eval/runs",
        json={"evalSuiteId": str(uuid4()), "checkpointId": str(uuid4())},
    )
    assert response.status_code == 501
    body = response.json()
    assert body["component"] == "eval"
    assert "not implemented" in body["detail"].lower()


def test_get_eval_result_returns_501(client: TestClient) -> None:
    response = client.get(f"/v1/eval/runs/{uuid4()}")
    assert response.status_code == 501
    assert response.json()["component"] == "eval"


def test_compare_checkpoints_returns_501(client: TestClient) -> None:
    response = client.post(
        "/v1/eval/compare",
        json={
            "evalSuiteId": str(uuid4()),
            "baselineCheckpointId": str(uuid4()),
            "candidateCheckpointId": str(uuid4()),
        },
    )
    assert response.status_code == 501
    body = response.json()
    assert body["component"] == "eval"
    assert "not implemented" in body["detail"].lower()
