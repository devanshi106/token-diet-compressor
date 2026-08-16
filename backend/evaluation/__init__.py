"""Evaluation package - offline benchmark utilities and EvalQuery dataclass."""
from backend.evaluation.evaluation import (
    AggregateReport,
    EvalQuery,
    ExperimentAResult,
    ExperimentBResult,
    aggregate_a,
    aggregate_b,
    run_experiment_a,
    run_experiment_b,
)

__all__ = [
    "AggregateReport",
    "EvalQuery",
    "ExperimentAResult",
    "ExperimentBResult",
    "aggregate_a",
    "aggregate_b",
    "run_experiment_a",
    "run_experiment_b",
]