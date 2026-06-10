"""Eval domain routes: eval suites + eval runs + checkpoint comparison.

v0.3 SKELETON. The real eval-execution engine is not implemented yet, so
the endpoints that need a running engine (start an eval run, read a run's
result, compare two checkpoints) return `501 Not Implemented` with a
standard `Problem` body. The endpoints whose answer doesn't depend on a
running engine are real:

  * `GET /v1/eval/suites` returns an empty list (no suites persisted yet).
  * `POST /v1/eval/suites` echoes back the created suite (the wire shape;
    real persistence lands with the engine).
  * `GET /v1/eval/suites/{id}` 404s (nothing persisted yet).
  * `DELETE /v1/eval/suites/{id}` 404s (nothing persisted yet).

When the engine lands it replaces the 501s and wires real suite storage;
the wire shapes here are the long-term contract.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request, Response, status

from .._generated.common_models import Problem
from .._generated.models import (
    EvalSuite,
    V1EvalComparePostRequest,
    V1EvalRunsPostRequest,
    V1EvalSuitesGetResponse,
)

router = APIRouter(tags=["eval"])

_ENGINE_NOT_IMPLEMENTED = (
    "eval execution engine not implemented in the v0.3 skeleton; "
    "this repo ships the control-plane wire shape only"
)


def _not_implemented(operation: str) -> Response:
    problem = Problem(
        type="https://github.com/eugene-plexus/eval#engine-not-implemented",
        title="Eval engine not implemented",
        status=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"{operation}: {_ENGINE_NOT_IMPLEMENTED}.",
        component="eval",
    )
    return Response(
        content=problem.model_dump_json(exclude_none=True),
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        media_type="application/problem+json",
    )


def _not_found(operation: str, resource_id: UUID) -> Response:
    problem = Problem(
        type="https://github.com/eugene-plexus/eval#not-found",
        title="Not found",
        status=status.HTTP_404_NOT_FOUND,
        detail=(
            f"{operation}: no such resource {resource_id} — the v0.3 skeleton "
            "persists no eval suites or runs yet."
        ),
        component="eval",
    )
    return Response(
        content=problem.model_dump_json(exclude_none=True),
        status_code=status.HTTP_404_NOT_FOUND,
        media_type="application/problem+json",
    )


# --------------------------------------------------------------------------- #
# Eval suites
# --------------------------------------------------------------------------- #


@router.get("/v1/eval/suites", response_model=V1EvalSuitesGetResponse)
async def list_eval_suites(request: Request) -> V1EvalSuitesGetResponse:
    """List eval suites. The skeleton persists no suites yet — returns an
    empty list rather than 501 so callers polling for suites get a valid
    empty result."""
    return V1EvalSuitesGetResponse(suites=[])


@router.post("/v1/eval/suites", response_model=EvalSuite, status_code=status.HTTP_201_CREATED)
async def create_eval_suite(request: Request, body: EvalSuite) -> EvalSuite:
    """Create an eval suite. The skeleton has no persistence layer, so it
    validates and echoes the suite back in the long-term wire shape; the
    engine work adds storage."""
    return body


@router.get("/v1/eval/suites/{eval_suite_id}", response_model=EvalSuite)
async def get_eval_suite(request: Request, eval_suite_id: UUID) -> Response:
    return _not_found("getEvalSuite", eval_suite_id)


@router.delete("/v1/eval/suites/{eval_suite_id}", status_code=204)
async def delete_eval_suite(request: Request, eval_suite_id: UUID) -> Response:
    return _not_found("deleteEvalSuite", eval_suite_id)


# --------------------------------------------------------------------------- #
# Eval runs + compare (engine-dependent: 501 in the skeleton)
# --------------------------------------------------------------------------- #


@router.post("/v1/eval/runs", status_code=202)
async def start_eval_run(request: Request, body: V1EvalRunsPostRequest) -> Response:
    return _not_implemented("startEvalRun")


@router.get("/v1/eval/runs/{eval_run_id}")
async def get_eval_result(request: Request, eval_run_id: UUID) -> Response:
    return _not_implemented("getEvalResult")


@router.post("/v1/eval/compare")
async def compare_checkpoints(request: Request, body: V1EvalComparePostRequest) -> Response:
    return _not_implemented("compareCheckpoints")
