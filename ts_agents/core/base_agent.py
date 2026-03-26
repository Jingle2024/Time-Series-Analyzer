"""
core/base_agent.py
──────────────────
Abstract base class every specialized agent inherits from.
Provides shared logging, timing, status tracking, and the
standard execute() / _run() lifecycle.
"""

from __future__ import annotations

import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AgentResult:
    """Standard result envelope returned by every agent."""
    agent_name: str
    status: AgentStatus
    data: Any = None                        # primary output payload
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == AgentStatus.SUCCESS


class BaseAgent(ABC):
    """
    Every agent follows the same lifecycle:
      1. validate_inputs()   – raise ValueError if required inputs missing
      2. _run()              – core logic, returns AgentResult
      3. execute()           – wraps _run() with timing & error handling
    """

    def __init__(self, name: str, context_store: Optional["ContextStore"] = None):
        self.name = name
        self.context_store = context_store
        self.status = AgentStatus.IDLE
        self.logger = logging.getLogger(f"agents.{name}")

    # ── public interface ──────────────────────────────────────────────────────

    def execute(self, **kwargs) -> AgentResult:
        """Run the agent. Catches exceptions, records timing."""
        self.status = AgentStatus.RUNNING
        start = time.perf_counter()
        try:
            self.validate_inputs(**kwargs)
            result = self._run(**kwargs)
            result.elapsed_seconds = time.perf_counter() - start
            self.status = result.status
            if self.context_store:
                self.context_store.set(self.name, result)
            self.logger.info(
                "Agent %-30s %-8s  %.2fs",
                self.name, result.status.value, result.elapsed_seconds,
            )
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - start
            self.status = AgentStatus.FAILED
            self.logger.error("Agent %s FAILED: %s", self.name, exc)
            result = AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=[str(exc), traceback.format_exc()],
                elapsed_seconds=elapsed,
            )
            if self.context_store:
                self.context_store.set(self.name, result)
            return result

    # ── override these ────────────────────────────────────────────────────────

    def validate_inputs(self, **kwargs) -> None:
        """Override to add input validation. Raise ValueError on failure."""

    @abstractmethod
    def _run(self, **kwargs) -> AgentResult:
        """Core agent logic. Must return an AgentResult."""
