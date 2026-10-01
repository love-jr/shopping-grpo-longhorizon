"""Versioned contracts for this repository's offline evaluator."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy

CONTRACT_VERSION = "shopping-trajectory-evaluation-v1"
RUBRIC_SCHEMA_VERSION = "shopping-requirement-rubric-v1"
JUDGE_SCHEMA_VERSION = "shopping-trajectory-judge-v1"

JUDGE_DIMENSIONS = (
    "search_strategy",
    "candidate_utilization",
    "evidence_verification",
    "decision_quality",
    "termination_efficiency",
)
RUBRIC_STATUSES = frozenset(
    {"satisfied", "violated", "unknown", "not_applicable"}
)
RUBRIC_HARDNESS = frozenset({"hard", "soft", "needs_review"})
JUDGE_STATUSES = frozenset({"valid", "invalid", "not_judged"})
ERROR_TAXONOMY = frozenset(
    {
        "search_core_requirement_missed",
        "search_ineffective_reformulation",
        "search_mechanical_repeat",
        "promising_candidate_ignored",
        "candidate_comparison_insufficient",
        "candidate_overexploration",
        "critical_evidence_missing",
        "unreliable_evidence_used",
        "final_price_unverified",
        "wrong_category",
        "budget_violation",
        "requirement_violation",
        "wrong_option",
        "premature_purchase",
        "premature_abstain",
        "unjustified_abstain",
        "overexploration_after_convergence",
        "repeat_loop",
        "max_steps_exhaustion",
        "illegal_action",
        "context_loss",
        "reward_rubric_disagreement",
        "infrastructure_invalid",
        "other",
    }
)


class ContractValidationError(ValueError):
    """Raised when a cached evaluator artifact violates its frozen schema."""


def _mapping(value: object, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{path} must be an object")
    return value


def _list(value: object, path: str) -> list:
    if not isinstance(value, list):
        raise ContractValidationError(f"{path} must be a list")
    return value


def _nonempty_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{path} must be a non-empty string")
    return value.strip()


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool):
        raise ContractValidationError(f"{path} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{path} must be an integer") from exc
    if value != integer:
        raise ContractValidationError(f"{path} must be an integer")
    return integer


def _unique_nonempty_strings(values: object, path: str) -> list[str]:
    result = []
    for index, value in enumerate(_list(values, path)):
        text = _nonempty_text(value, f"{path}[{index}]")
        if text in result:
            raise ContractValidationError(f"{path} contains duplicate {text!r}")
        result.append(text)
    return result


def validate_rubric_bundle(
    bundle: object,
    *,
    expected_task_id: int | None = None,
) -> dict:
    """Validate and defensively copy one frozen task-level Rubric bundle."""

    payload = _mapping(bundle, "rubric_bundle")
    if payload.get("schema_version") != RUBRIC_SCHEMA_VERSION:
        raise ContractValidationError(
            "rubric_bundle.schema_version must be "
            f"{RUBRIC_SCHEMA_VERSION!r}"
        )
    task_id = _integer(payload.get("task_id"), "rubric_bundle.task_id")
    if expected_task_id is not None and task_id != int(expected_task_id):
        raise ContractValidationError(
            f"rubric task_id {task_id} does not match {int(expected_task_id)}"
        )
    query = _nonempty_text(payload.get("query"), "rubric_bundle.query")
    _nonempty_text(
        payload.get("rubric_version"), "rubric_bundle.rubric_version"
    )
    generation = _mapping(
        payload.get("generation"), "rubric_bundle.generation"
    )
    for field in (
        "extractor_version",
        "curator_model",
        "curator_prompt_version",
        "task_data_hash",
        "query_hash",
    ):
        _nonempty_text(generation.get(field), f"rubric_bundle.generation.{field}")

    rubric_ids = set()
    candidate_ids = set()
    rubrics = _list(payload.get("rubrics"), "rubric_bundle.rubrics")
    if not rubrics:
        raise ContractValidationError("rubric_bundle.rubrics must not be empty")
    for index, item_value in enumerate(rubrics):
        path = f"rubric_bundle.rubrics[{index}]"
        item = _mapping(item_value, path)
        rubric_id = _nonempty_text(item.get("rubric_id"), f"{path}.rubric_id")
        candidate_id = _nonempty_text(
            item.get("candidate_id"), f"{path}.candidate_id"
        )
        if rubric_id in rubric_ids:
            raise ContractValidationError(f"duplicate rubric_id {rubric_id!r}")
        if candidate_id in candidate_ids:
            raise ContractValidationError(
                f"duplicate selected candidate_id {candidate_id!r}"
            )
        rubric_ids.add(rubric_id)
        candidate_ids.add(candidate_id)
        _nonempty_text(
            item.get("constraint_type"), f"{path}.constraint_type"
        )
        _nonempty_text(item.get("description"), f"{path}.description")
        hardness = _nonempty_text(item.get("hardness"), f"{path}.hardness")
        if hardness not in RUBRIC_HARDNESS:
            raise ContractValidationError(
                f"{path}.hardness must be one of {sorted(RUBRIC_HARDNESS)}"
            )
        _nonempty_text(
            item.get("hardness_source"), f"{path}.hardness_source"
        )
        _nonempty_text(item.get("field_path"), f"{path}.field_path")
        _nonempty_text(item.get("operator"), f"{path}.operator")
        if "expected_value" not in item:
            raise ContractValidationError(f"{path}.expected_value is required")
        _unique_nonempty_strings(item.get("data_sources"), f"{path}.data_sources")
        spans = _list(item.get("query_spans"), f"{path}.query_spans")
        if not spans:
            raise ContractValidationError(f"{path}.query_spans must not be empty")
        for span_index, span_value in enumerate(spans):
            span_path = f"{path}.query_spans[{span_index}]"
            span = _mapping(span_value, span_path)
            _nonempty_text(span.get("text"), f"{span_path}.text")
            start = _integer(span.get("start"), f"{span_path}.start")
            end = _integer(span.get("end"), f"{span_path}.end")
            if start < 0 or end <= start:
                raise ContractValidationError(
                    f"{span_path} must satisfy 0 <= start < end"
                )
            if end > len(query) or query[start:end] != span["text"]:
                raise ContractValidationError(
                    f"{span_path} does not match rubric_bundle.query"
                )

    return deepcopy(dict(payload))


def validate_curator_response(
    response: object,
    *,
    candidate_ids: Iterable[str],
    query: str,
) -> dict:
    """Ensure Flash selects only supplied candidates with direct Query evidence."""

    payload = _mapping(response, "curator_response")
    allowed = {str(candidate_id) for candidate_id in candidate_ids}
    selected = _list(
        payload.get("selected_constraints"),
        "curator_response.selected_constraints",
    )
    if not selected:
        raise ContractValidationError("curator_response.selected_constraints must not be empty")
    seen = set()
    for index, item_value in enumerate(selected):
        path = f"curator_response.selected_constraints[{index}]"
        item = _mapping(item_value, path)
        candidate_id = _nonempty_text(
            item.get("candidate_id"), f"{path}.candidate_id"
        )
        if candidate_id not in allowed:
            raise ContractValidationError(
                f"{path} references unknown candidate_id {candidate_id!r}"
            )
        if candidate_id in seen:
            raise ContractValidationError(
                f"{path} repeats candidate_id {candidate_id!r}"
            )
        seen.add(candidate_id)
        _nonempty_text(item.get("description"), f"{path}.description")
        hardness = _nonempty_text(item.get("hardness"), f"{path}.hardness")
        if hardness not in RUBRIC_HARDNESS:
            raise ContractValidationError(
                f"{path}.hardness must be one of {sorted(RUBRIC_HARDNESS)}"
            )
        quote = _nonempty_text(item.get("query_quote"), f"{path}.query_quote")
        if quote not in query:
            raise ContractValidationError(
                f"{path}.query_quote must be a literal substring of the task query"
            )
        _nonempty_text(
            item.get("selection_reason"), f"{path}.selection_reason"
        )
    return deepcopy(dict(payload))


def validate_judge_result(
    result: object,
    *,
    rubric_ids: Iterable[str],
    expected_task_id: int | None = None,
    expected_trajectory_id: str | None = None,
    allowed_event_ids: Iterable[str] | None = None,
) -> dict:
    """Validate one Flash Judge result without manufacturing missing scores."""

    payload = _mapping(result, "judge_result")
    if payload.get("schema_version") != JUDGE_SCHEMA_VERSION:
        raise ContractValidationError(
            f"judge_result.schema_version must be {JUDGE_SCHEMA_VERSION!r}"
        )
    for forbidden in ("total_score", "overall_score", "weighted_total"):
        if forbidden in payload:
            raise ContractValidationError(
                f"judge_result must not contain {forbidden}"
            )
    task_id = _integer(payload.get("task_id"), "judge_result.task_id")
    trajectory_id = _nonempty_text(
        payload.get("trajectory_id"), "judge_result.trajectory_id"
    )
    if expected_task_id is not None and task_id != int(expected_task_id):
        raise ContractValidationError(
            f"judge task_id {task_id} does not match {int(expected_task_id)}"
        )
    if (
        expected_trajectory_id is not None
        and trajectory_id != str(expected_trajectory_id)
    ):
        raise ContractValidationError(
            "judge trajectory_id does not match the evaluated trajectory"
        )
    judge_status = _nonempty_text(
        payload.get("judge_status"), "judge_result.judge_status"
    )
    if judge_status not in JUDGE_STATUSES:
        raise ContractValidationError(
            f"judge_status must be one of {sorted(JUDGE_STATUSES)}"
        )
    allowed_events = (
        {str(event_id) for event_id in allowed_event_ids}
        if allowed_event_ids is not None
        else None
    )

    def validate_evidence(value: object, path: str) -> list[str]:
        event_ids = _unique_nonempty_strings(value, path)
        if allowed_events is not None:
            unknown = sorted(set(event_ids) - allowed_events)
            if unknown:
                raise ContractValidationError(
                    f"{path} references unknown event IDs: {unknown}"
                )
        return event_ids

    expected_rubrics = {str(rubric_id) for rubric_id in rubric_ids}
    assessments = _list(
        payload.get("rubric_assessments"),
        "judge_result.rubric_assessments",
    )
    dimensions = _mapping(
        payload.get("dimension_scores"), "judge_result.dimension_scores"
    )
    if judge_status == "valid":
        seen_rubrics = set()
        for index, assessment_value in enumerate(assessments):
            path = f"judge_result.rubric_assessments[{index}]"
            assessment = _mapping(assessment_value, path)
            rubric_id = _nonempty_text(
                assessment.get("rubric_id"), f"{path}.rubric_id"
            )
            if rubric_id not in expected_rubrics:
                raise ContractValidationError(
                    f"{path} references unknown rubric_id {rubric_id!r}"
                )
            if rubric_id in seen_rubrics:
                raise ContractValidationError(
                    f"{path} repeats rubric_id {rubric_id!r}"
                )
            seen_rubrics.add(rubric_id)
            status = _nonempty_text(
                assessment.get("status"), f"{path}.status"
            )
            if status not in RUBRIC_STATUSES:
                raise ContractValidationError(
                    f"{path}.status must be one of {sorted(RUBRIC_STATUSES)}"
                )
            _nonempty_text(assessment.get("reason"), f"{path}.reason")
            validate_evidence(
                assessment.get("evidence_event_ids"),
                f"{path}.evidence_event_ids",
            )
        if seen_rubrics != expected_rubrics:
            missing = sorted(expected_rubrics - seen_rubrics)
            raise ContractValidationError(
                f"judge_result is missing rubric assessments: {missing}"
            )
        if set(dimensions) != set(JUDGE_DIMENSIONS):
            raise ContractValidationError(
                "dimension_scores must contain exactly "
                f"{list(JUDGE_DIMENSIONS)}"
            )
        for name in JUDGE_DIMENSIONS:
            path = f"judge_result.dimension_scores.{name}"
            score_payload = _mapping(dimensions[name], path)
            score = _integer(score_payload.get("score"), f"{path}.score")
            if score not in {0, 1, 2}:
                raise ContractValidationError(f"{path}.score must be 0, 1, or 2")
            _nonempty_text(score_payload.get("reason"), f"{path}.reason")
            validate_evidence(
                score_payload.get("evidence_event_ids"),
                f"{path}.evidence_event_ids",
            )
    else:
        if assessments or dimensions:
            raise ContractValidationError(
                "invalid/not_judged results must not contain inferred assessments "
                "or dimension scores"
            )

    errors = _mapping(payload.get("errors"), "judge_result.errors")
    primary = errors.get("primary")
    if primary is not None and not isinstance(primary, str):
        raise ContractValidationError(
            "judge_result.errors.primary must be a string or null"
        )
    if primary is not None and primary not in ERROR_TAXONOMY:
        raise ContractValidationError(
            "judge_result.errors.primary is not in the frozen taxonomy"
        )
    secondary = _unique_nonempty_strings(
        errors.get("secondary"), "judge_result.errors.secondary"
    )
    if len(secondary) > 2:
        raise ContractValidationError(
            "judge_result.errors.secondary must contain at most two values"
        )
    unknown_secondary = sorted(set(secondary) - ERROR_TAXONOMY)
    if unknown_secondary:
        raise ContractValidationError(
            "judge_result.errors.secondary contains unknown taxonomy values: "
            f"{unknown_secondary}"
        )
    if primary is not None and primary in secondary:
        raise ContractValidationError(
            "judge_result.errors.primary must not be repeated in secondary"
        )
    if primary is None and secondary:
        raise ContractValidationError(
            "judge_result.errors.secondary must be empty when primary is null"
        )
    validate_evidence(
        errors.get("evidence_event_ids"),
        "judge_result.errors.evidence_event_ids",
    )
    diagnosis = payload.get("overall_diagnosis")
    if not isinstance(diagnosis, str):
        raise ContractValidationError(
            "judge_result.overall_diagnosis must be a string"
        )
    return deepcopy(dict(payload))


def rubric_ids(bundle: Mapping) -> list[str]:
    """Return rubric IDs after validating the bundle."""

    validated = validate_rubric_bundle(bundle)
    return [item["rubric_id"] for item in validated["rubrics"]]
