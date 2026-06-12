"""Tests for the eval domain routes against the live engine.

Covers suite CRUD (persisted), eval runs (val-loss/perplexity over a validation
dataset; sample-review/token-entropy over prompts), result fetch, checkpoint
comparison, and the error mappings.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from fastapi.testclient import TestClient


def _create_suite(client: TestClient, **fields: object) -> str:
    suite_id = str(uuid4())
    body = {"evalSuiteId": suite_id, "name": "suite", **fields}
    resp = client.post("/v1/eval/suites", json=body)
    assert resp.status_code == 201, resp.text
    return suite_id


def test_list_eval_suites_starts_empty(client: TestClient) -> None:
    response = client.get("/v1/eval/suites")
    assert response.status_code == 200
    assert response.json() == {"suites": []}


def test_suite_crud_round_trip(client: TestClient) -> None:
    suite_id = _create_suite(client, name="smoke", metrics=["perplexity"])
    # listed + fetchable
    assert any(s["evalSuiteId"] == suite_id for s in client.get("/v1/eval/suites").json()["suites"])
    got = client.get(f"/v1/eval/suites/{suite_id}")
    assert got.status_code == 200
    assert got.json()["name"] == "smoke"
    # deletable, then gone
    assert client.delete(f"/v1/eval/suites/{suite_id}").status_code == 204
    assert client.get(f"/v1/eval/suites/{suite_id}").status_code == 404


def test_create_duplicate_suite_409(client: TestClient) -> None:
    suite_id = str(uuid4())
    body = {"evalSuiteId": suite_id, "name": "dup", "metrics": ["perplexity"]}
    assert client.post("/v1/eval/suites", json=body).status_code == 201
    assert client.post("/v1/eval/suites", json=body).status_code == 409


def test_create_eval_suite_rejects_bad_metric(client: TestClient) -> None:
    body = {"evalSuiteId": str(uuid4()), "name": "bad", "metrics": ["not_a_real_metric"]}
    assert client.post("/v1/eval/suites", json=body).status_code == 422


def test_get_unknown_suite_404(client: TestClient) -> None:
    response = client.get(f"/v1/eval/suites/{uuid4()}")
    assert response.status_code == 404
    assert response.json()["component"] == "eval"


def test_run_reports_val_loss_and_perplexity(
    client: TestClient,
    make_arithmetic_checkpoint: Callable[..., str],
    make_arithmetic_dataset: Callable[..., str],
) -> None:
    dataset_id = make_arithmetic_dataset(vocab=32)
    checkpoint_id = make_arithmetic_checkpoint(vocab=32, steps=200)
    suite_id = _create_suite(
        client, validationDatasetId=dataset_id, metrics=["val_loss", "perplexity"]
    )

    resp = client.post(
        "/v1/eval/runs", json={"evalSuiteId": suite_id, "checkpointId": checkpoint_id}
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    assert isinstance(body["valLoss"], int | float)
    assert body["perplexity"] >= 1.0  # exp(cross-entropy >= 0)
    assert "samples" not in body  # not requested -> omitted


def test_run_sample_review_and_token_entropy(
    client: TestClient, make_text_checkpoint: Callable[..., str]
) -> None:
    checkpoint_id = make_text_checkpoint()
    suite_id = _create_suite(
        client,
        prompts=["hello there", "the answer is"],
        metrics=["sample_review", "token_entropy"],
    )

    resp = client.post(
        "/v1/eval/runs", json={"evalSuiteId": suite_id, "checkpointId": checkpoint_id}
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    assert len(body["samples"]) == 2
    assert all(set(s) >= {"prompt", "response"} for s in body["samples"])
    assert isinstance(body["tokenEntropy"], int | float)
    assert body["tokenEntropy"] >= 0.0


def test_get_result_after_run(
    client: TestClient,
    make_arithmetic_checkpoint: Callable[..., str],
    make_arithmetic_dataset: Callable[..., str],
) -> None:
    dataset_id = make_arithmetic_dataset(vocab=32)
    checkpoint_id = make_arithmetic_checkpoint(vocab=32, steps=100)
    suite_id = _create_suite(client, validationDatasetId=dataset_id, metrics=["val_loss"])
    run = client.post(
        "/v1/eval/runs", json={"evalSuiteId": suite_id, "checkpointId": checkpoint_id}
    ).json()

    fetched = client.get(f"/v1/eval/runs/{run['evalRunId']}")
    assert fetched.status_code == 200
    assert fetched.json()["valLoss"] == run["valLoss"]


def test_compare_trained_beats_untrained(
    client: TestClient,
    make_arithmetic_checkpoint: Callable[..., str],
    make_arithmetic_dataset: Callable[..., str],
) -> None:
    dataset_id = make_arithmetic_dataset(vocab=32)
    untrained = make_arithmetic_checkpoint(vocab=32, steps=0)
    trained = make_arithmetic_checkpoint(vocab=32, steps=250)
    suite_id = _create_suite(client, validationDatasetId=dataset_id, metrics=["val_loss"])

    resp = client.post(
        "/v1/eval/compare",
        json={
            "evalSuiteId": suite_id,
            "baselineCheckpointId": untrained,
            "candidateCheckpointId": trained,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The seeded validation loader feeds both runs the same batches, so the
    # trained checkpoint must score a strictly lower val-loss.
    assert body["candidate"]["valLoss"] < body["baseline"]["valLoss"]


def test_run_perplexity_only_omits_val_loss(
    client: TestClient,
    make_arithmetic_checkpoint: Callable[..., str],
    make_arithmetic_dataset: Callable[..., str],
) -> None:
    dataset_id = make_arithmetic_dataset(vocab=32)
    checkpoint_id = make_arithmetic_checkpoint(vocab=32, steps=50)
    suite_id = _create_suite(client, validationDatasetId=dataset_id, metrics=["perplexity"])
    body = client.post(
        "/v1/eval/runs", json={"evalSuiteId": suite_id, "checkpointId": checkpoint_id}
    ).json()
    assert "perplexity" in body
    assert "valLoss" not in body  # computed internally but not requested -> omitted


def test_run_token_entropy_only_omits_samples(
    client: TestClient, make_text_checkpoint: Callable[..., str]
) -> None:
    checkpoint_id = make_text_checkpoint()
    suite_id = _create_suite(client, prompts=["hello"], metrics=["token_entropy"])
    body = client.post(
        "/v1/eval/runs", json={"evalSuiteId": suite_id, "checkpointId": checkpoint_id}
    ).json()
    assert "tokenEntropy" in body
    assert "samples" not in body  # generated internally but not requested -> omitted


def test_run_val_loss_without_dataset_400(
    client: TestClient, make_arithmetic_checkpoint: Callable[..., str]
) -> None:
    checkpoint_id = make_arithmetic_checkpoint(vocab=32, steps=0)
    suite_id = _create_suite(client, metrics=["val_loss"])  # no validationDatasetId
    resp = client.post(
        "/v1/eval/runs", json={"evalSuiteId": suite_id, "checkpointId": checkpoint_id}
    )
    assert resp.status_code == 400
    assert "validationdatasetid" in resp.json()["detail"].lower().replace(" ", "")


def test_run_sample_review_without_prompts_400(
    client: TestClient, make_text_checkpoint: Callable[..., str]
) -> None:
    checkpoint_id = make_text_checkpoint()
    suite_id = _create_suite(client, metrics=["sample_review"])  # no prompts
    resp = client.post(
        "/v1/eval/runs", json={"evalSuiteId": suite_id, "checkpointId": checkpoint_id}
    )
    assert resp.status_code == 400
    assert "prompts" in resp.json()["detail"].lower()


def test_run_sample_review_without_tokenizer_400(
    client: TestClient, make_arithmetic_checkpoint: Callable[..., str]
) -> None:
    # An arithmetic checkpoint has no embedded tokenizer -> can't generate text.
    checkpoint_id = make_arithmetic_checkpoint(vocab=32, steps=0)
    suite_id = _create_suite(client, prompts=["hi"], metrics=["sample_review"])
    resp = client.post(
        "/v1/eval/runs", json={"evalSuiteId": suite_id, "checkpointId": checkpoint_id}
    )
    assert resp.status_code == 400
    assert "tokenizer" in resp.json()["detail"].lower()


def test_run_unknown_suite_404(client: TestClient) -> None:
    resp = client.post(
        "/v1/eval/runs", json={"evalSuiteId": str(uuid4()), "checkpointId": str(uuid4())}
    )
    assert resp.status_code == 404


def test_run_unknown_checkpoint_404(
    client: TestClient, make_arithmetic_dataset: Callable[..., str]
) -> None:
    suite_id = _create_suite(
        client, validationDatasetId=make_arithmetic_dataset(), metrics=["val_loss"]
    )
    resp = client.post(
        "/v1/eval/runs", json={"evalSuiteId": suite_id, "checkpointId": str(uuid4())}
    )
    assert resp.status_code == 404


def test_get_unknown_run_404(client: TestClient) -> None:
    assert client.get(f"/v1/eval/runs/{uuid4()}").status_code == 404
