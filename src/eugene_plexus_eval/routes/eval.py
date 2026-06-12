"""Eval domain routes: eval suites (persisted CRUD) + eval runs + compare.

The run/compare handlers are plain ``def`` (not ``async``) so FastAPI runs the
blocking eval (model forward + generation) in a worker thread. Suite reads
answer with empty/own shapes even when the engine is unavailable (safe mode /
degraded); the executing endpoints return ``503`` with a ``Problem`` body.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import JSONResponse

from .._generated.common_models import Problem
from .._generated.models import (
    EvalSuite,
    V1EvalComparePostRequest,
    V1EvalComparePostResponse,
    V1EvalRunsPostRequest,
    V1EvalSuitesGetResponse,
)
from ..engine.engine import BadRequestError, ConflictError, EvalEngine, EvalError, NotFoundError

router = APIRouter(tags=["eval"])

_ERR_STATUS: list[tuple[type[EvalError], int]] = [
    (NotFoundError, status.HTTP_404_NOT_FOUND),
    (ConflictError, status.HTTP_409_CONFLICT),
    (BadRequestError, status.HTTP_400_BAD_REQUEST),
]


def _problem(status_code: int, title: str, detail: str) -> JSONResponse:
    slug = title.replace(" ", "-").lower()
    body = Problem(
        type=f"https://github.com/eugene-plexus/eval#{slug}",
        title=title,
        status=status_code,
        detail=detail,
        component="eval",
    )
    return JSONResponse(
        status_code=status_code,
        media_type="application/problem+json",
        content=body.model_dump(exclude_none=True),
    )


def _engine_error(e: EvalError) -> JSONResponse:
    code = next(
        (c for cls, c in _ERR_STATUS if isinstance(e, cls)), status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    return _problem(code, type(e).__name__, str(e))


def _engine(request: Request) -> EvalEngine | None:
    return getattr(request.app.state, "engine", None)


def _unavailable(request: Request) -> JSONResponse:
    detail = getattr(request.app.state, "engine_error", None) or "eval is unavailable"
    return _problem(status.HTTP_503_SERVICE_UNAVAILABLE, "Eval unavailable", detail)


# --------------------------------------------------------------------------- #
# Eval suites
# --------------------------------------------------------------------------- #


@router.get("/v1/eval/suites", response_model=V1EvalSuitesGetResponse)
def list_eval_suites(request: Request) -> V1EvalSuitesGetResponse:
    engine = _engine(request)
    suites = engine.list_suites() if engine else []
    return V1EvalSuitesGetResponse(suites=suites)


@router.post("/v1/eval/suites", status_code=status.HTTP_201_CREATED)
def create_eval_suite(request: Request, body: EvalSuite) -> Response:
    engine = _engine(request)
    if engine is None:
        return _unavailable(request)
    try:
        suite = engine.create_suite(body)
    except EvalError as e:
        return _engine_error(e)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content=suite.model_dump(mode="json", exclude_none=True),
    )


@router.get("/v1/eval/suites/{eval_suite_id}")
def get_eval_suite(request: Request, eval_suite_id: UUID) -> Response:
    engine = _engine(request)
    if engine is None:
        return _unavailable(request)
    try:
        suite = engine.get_suite(eval_suite_id)
    except EvalError as e:
        return _engine_error(e)
    return JSONResponse(content=suite.model_dump(mode="json", exclude_none=True))


@router.delete("/v1/eval/suites/{eval_suite_id}", status_code=204)
def delete_eval_suite(request: Request, eval_suite_id: UUID) -> Response:
    engine = _engine(request)
    if engine is None:
        return _unavailable(request)
    try:
        engine.delete_suite(eval_suite_id)
    except EvalError as e:
        return _engine_error(e)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --------------------------------------------------------------------------- #
# Eval runs + compare
# --------------------------------------------------------------------------- #


@router.post("/v1/eval/runs", status_code=status.HTTP_202_ACCEPTED)
def start_eval_run(request: Request, body: V1EvalRunsPostRequest) -> Response:
    engine = _engine(request)
    if engine is None:
        return _unavailable(request)
    try:
        result = engine.run_eval(eval_suite_id=body.evalSuiteId, checkpoint_id=body.checkpointId)
    except EvalError as e:
        return _engine_error(e)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=result.model_dump(mode="json", exclude_none=True),
    )


@router.get("/v1/eval/runs/{eval_run_id}")
def get_eval_result(request: Request, eval_run_id: UUID) -> Response:
    engine = _engine(request)
    if engine is None:
        return _unavailable(request)
    try:
        result = engine.get_result(eval_run_id)
    except EvalError as e:
        return _engine_error(e)
    return JSONResponse(content=result.model_dump(mode="json", exclude_none=True))


@router.post("/v1/eval/compare")
def compare_checkpoints(request: Request, body: V1EvalComparePostRequest) -> Response:
    engine = _engine(request)
    if engine is None:
        return _unavailable(request)
    try:
        baseline, candidate = engine.compare(
            eval_suite_id=body.evalSuiteId,
            baseline_checkpoint_id=body.baselineCheckpointId,
            candidate_checkpoint_id=body.candidateCheckpointId,
        )
    except EvalError as e:
        return _engine_error(e)
    response = V1EvalComparePostResponse(baseline=baseline, candidate=candidate)
    return JSONResponse(content=response.model_dump(mode="json", exclude_none=True))
