from dataclasses import dataclass, field
from typing import Any, Dict
from datetime import datetime


@dataclass
class Trace:
    """
    A pheromone trail left by an agent.
    """

    agent: str
    task: str
    type: str
    data: Dict[str, Any]
    time: datetime = field(default_factory=datetime.now)
    strength: float = 1.0
    invalidated: bool = False

    def decay(self, rate: float = 0.15) -> bool:
        """Weaken this trace. Returns False when too weak."""
        self.strength *= 1 - rate
        return self.strength > 0.05

    def invalidate(self):
        """Mark trace as invalid, triggering re-computation."""
        self.invalidated = True

    @property
    def age(self) -> float:
        """Seconds since trace was created."""
        return (datetime.now() - self.time).total_seconds()

    @property
    def alive(self) -> bool:
        """Trace is usable if strong enough and not invalidated."""
        return self.strength > 0.05 and not self.invalidated
