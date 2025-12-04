from typing import Any, Dict, List, Optional
from collections import defaultdict
import asyncio
import hashlib

from .trace import Trace


class Environment:
    """
    Shared memory where agents communicate via traces (stigmergy).
    Each task has isolated traces. Thread-safe for concurrent task execution.
    """

    def __init__(self):
        self._traces: Dict[str, List[Trace]] = defaultdict(list)
        self._write_lock = asyncio.Lock()

    async def add(
        self, agent: str, task: str, type: str, data: Dict[str, Any]
    ) -> Trace:
        """Agent leaves a trace. Returns the created trace."""
        trace = Trace(agent=agent, task=task, type=type, data=data)
        async with self._write_lock:
            self._traces[task].append(trace)
        return trace

    def add_sync(self, agent: str, task: str, type: str, data: Dict[str, Any]) -> Trace:
        """Synchronous version for initialization (not thread-safe)."""
        trace = Trace(agent=agent, task=task, type=type, data=data)
        self._traces[task].append(trace)
        return trace

    def read(self, task: str, type: Optional[str] = None) -> List[Trace]:
        """Read alive traces, optionally filtered by type."""
        traces = [t for t in self._traces[task] if t.alive]
        if type:
            traces = [t for t in traces if t.type == type]
        return sorted(traces, key=lambda t: t.time)

    def strongest(self, task: str, type: str) -> Optional[Trace]:
        """Get the strongest alive trace of a given type."""
        traces = self.read(task, type)
        return max(traces, key=lambda t: t.strength) if traces else None

    def freshest(self, task: str, type: str) -> Optional[Trace]:
        """Get the most recent alive trace of a given type."""
        traces = self.read(task, type)
        return min(traces, key=lambda t: t.age) if traces else None

    def strongest_data(self, task: str, type: str) -> Optional[Dict[str, Any]]:
        """Get data from the strongest alive trace, or None."""
        trace = self.strongest(task, type)
        return trace.data if trace else None

    def freshest_data(self, task: str, type: str) -> Optional[Dict[str, Any]]:
        """Get data from the most recent alive trace, or None."""
        trace = self.freshest(task, type)
        return trace.data if trace else None

    def invalidate(
        self, task: str, agent: Optional[str] = None, type: Optional[str] = None
    ):
        """
        Mark traces as invalid, triggering re-computation.
        Can filter by agent, type, or both.
        """
        for trace in self._traces[task]:
            if agent and trace.agent != agent:
                continue
            if type and trace.type != type:
                continue
            trace.invalidate()

    def reinforce(self, task: str, type: str, amount: float = 0.3):
        """Strengthen traces of a type (positive feedback)."""
        for trace in self._traces[task]:
            if trace.type == type and trace.alive:
                trace.strength = min(1.0, trace.strength + amount)

    def decay(self, task: str, rates: Optional[Dict[str, float]] = None):
        """Evaporate pheromones (sync version). Can specify custom rates per type."""
        default_rate = 0.15
        for trace in self._traces[task]:
            rate = rates.get(trace.type, default_rate) if rates else default_rate
            trace.decay(rate)
        self._traces[task] = [t for t in self._traces[task] if t.alive]

    async def decay_async(self, task: str, rates: Optional[Dict[str, float]] = None):
        """Evaporate pheromones (async/thread-safe version)."""
        async with self._write_lock:
            self.decay(task, rates)

    def snapshot(self, task: str) -> str:
        """Hash of current state for cycle detection."""
        traces = self.read(task)
        state = tuple((t.type, t.agent, round(t.strength, 1)) for t in traces)
        return hashlib.md5(str(state).encode()).hexdigest()

    def has_errors(self, task: str) -> bool:
        """Check if any error traces exist for this task."""
        return bool(self.read(task, "error"))

    def errors(self, task: str) -> List[Dict[str, Any]]:
        """Get all error trace data for this task."""
        return [t.data for t in self.read(task, "error")]

    def clear(self, task: str):
        """Remove all traces for a task."""
        self._traces[task] = []

    def observe(self, task: str) -> str:
        """Debug view of all traces for a task, sorted by time."""
        traces = sorted(self._traces[task], key=lambda t: t.time)
        if not traces:
            return "(no traces)"

        lines = []
        for t in traces:
            status = "✓" if t.alive else "✗"
            lines.append(
                f"{status} [{t.type}] {t.agent} "
                f"(str={t.strength:.2f}, age={t.age:.1f}s)"
            )
        return "\n".join(lines)
