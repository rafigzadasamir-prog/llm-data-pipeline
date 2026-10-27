"""A small, hand-rolled sequential pipeline runner.

This intentionally is not Airflow/Prefect/Dagster. The pipeline here is a
straight-line chain of five dependent stages (ingest -> extract -> validate
-> store -> report) with no branching, retries-across-stages, or scheduling
needs - a class that calls functions in order and records how long each one
took is a complete solution, and it keeps the dependency footprint at zero.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageTiming:
    """Timing and outcome for a single pipeline stage."""

    name: str
    duration_seconds: float
    status: str  # "ok" or "failed"
    error: str | None = None


class PipelineStageError(RuntimeError):
    """Raised when a stage function raises; wraps the original exception."""

    def __init__(self, stage_name: str, original: Exception) -> None:
        self.stage_name = stage_name
        self.original = original
        super().__init__(f"stage {stage_name!r} failed: {original}")


@dataclass
class PipelineRun:
    """Result of running a :class:`Pipeline`: the final value plus timings."""

    value: Any
    timings: list[StageTiming] = field(default_factory=list)


class Pipeline:
    """Runs a sequence of named stage functions, threading a value through them.

    Each stage is a ``Callable[[Any], Any]``: it receives the previous
    stage's return value and returns the value for the next stage. Stages
    run strictly in the order they were added - this is a straight line,
    not a DAG, because the ingest->extract->validate->store->report flow
    has no branches.
    """

    def __init__(self) -> None:
        self._stages: list[tuple[str, Callable[[Any], Any]]] = []

    def add_stage(self, name: str, func: Callable[[Any], Any]) -> Pipeline:
        self._stages.append((name, func))
        return self

    def run(self, initial_value: Any) -> PipelineRun:
        value = initial_value
        timings: list[StageTiming] = []
        for name, func in self._stages:
            start = time.perf_counter()
            try:
                value = func(value)
            except Exception as exc:
                duration = time.perf_counter() - start
                timings.append(StageTiming(name, duration, "failed", str(exc)))
                logger.exception("pipeline stage %r failed after %.3fs", name, duration)
                raise PipelineStageError(name, exc) from exc
            duration = time.perf_counter() - start
            timings.append(StageTiming(name, duration, "ok"))
            logger.info("pipeline stage %r completed in %.3fs", name, duration)
        return PipelineRun(value=value, timings=timings)
