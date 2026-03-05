"""Minimal type definitions used by google_d3pm_ins_del.training_setup."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class State:
    """Container mirroring the State shape expected by training_setup."""

    model_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any]
    params: Optional[Any]
    step: int

