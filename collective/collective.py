from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Optional, Set
from collections import defaultdict
import asyncio

from .agent import Agent
from .environment import Environment


@dataclass
class Result:
    """Result of a collective run."""

    task: str
    converged: bool
    rounds: int
    cycle_detected: bool
    traces: Dict[str, Any]
    errors: List[Dict[str, Any]]


class Collective:
    """
    Self-organizing collective of agents.

    Usage:
        collective = Collective()
        collective.add(agent1, agent2)
        result = await collective.run(task="my_task", goal="Generate UI")
    """

    def __init__(
        self,
        decay_rates: Optional[Dict[str, float]] = None,
        max_cycle_history: int = 10,
    ):
        self.agents: List[Agent] = []
        self.env = Environment()
        self.decay_rates = decay_rates or {}
        self.max_cycle_history = max_cycle_history

    def add(self, *agents: Agent) -> "Collective":
        """Add agents to the collective. Chainable."""
        self.agents.extend(agents)
        return self

    async def run(
        self,
        task: str,
        goal: str,
        goal_type: str = "goal",
        max_rounds: int = 10,
        on_round: Optional[Callable[[int, List[str]], None]] = None,
        stop_on_cycle: bool = True,
    ) -> Result:
        """
        Run the collective until convergence.

        Args:
            task: Unique task ID
            goal: The objective (stored in initial trace)
            goal_type: Type of the initial goal trace
            max_rounds: Maximum iterations
            on_round: Optional callback(round_num, active_agent_names)
            stop_on_cycle: Stop if state cycle detected

        Returns:
            RunResult with outcome and metadata
        """
        # Initialize
        self.env.add_sync("user", task, goal_type, {"goal": goal})

        state_history: List[str] = []
        errors: List[Dict[str, Any]] = []
        converged = False
        cycle_detected = False
        idle_rounds = 0
        round_num = 0

        for round_num in range(max_rounds):
            # Cycle detection
            current_state = self.env.snapshot(task)
            if stop_on_cycle and current_state in state_history:
                cycle_detected = True
                converged = True
                break
            state_history.append(current_state)
            if len(state_history) > self.max_cycle_history:
                state_history.pop(0)

            # Find activatable agents
            active = [a for a in self.agents if a.can_activate(self.env, task)]

            if on_round:
                on_round(round_num + 1, [a.name for a in active])

            if not active:
                idle_rounds += 1
                if idle_rounds >= 2:
                    converged = True
                    break
            else:
                idle_rounds = 0

                # Execute with individual error handling
                results = await asyncio.gather(
                    *[self._safe_execute(a, task) for a in active],
                    return_exceptions=False,
                )

                # Collect errors
                for agent, result in zip(active, results):
                    if isinstance(result, Exception):
                        errors.append(
                            {
                                "agent": agent.name,
                                "error": str(result),
                                "round": round_num + 1,
                            }
                        )

            # Decay (thread-safe)
            await self.env.decay_async(task, self.decay_rates)

        return Result(
            task=task,
            converged=converged,
            rounds=round_num + 1,
            cycle_detected=cycle_detected,
            traces=self._summarize_traces(task),
            errors=errors,
        )

    async def _safe_execute(self, agent: Agent, task: str) -> Any:
        """Execute agent with error handling. Leaves error trace on failure."""
        try:
            return await agent.act(self.env, task)
        except Exception as e:
            await self.env.add(
                agent=agent.name,
                task=task,
                type="error",
                data={"error": str(e), "agent": agent.name},
            )
            return e

    def result(self, task: str, type: str) -> Optional[Dict[str, Any]]:
        """Extract final result of a specific type."""
        return self.env.strongest_data(task, type)

    def results(self, task: str, *types: str) -> Dict[str, Optional[Dict[str, Any]]]:
        """Extract multiple result types at once."""
        return {t: self.result(task, t) for t in types}

    def _summarize_traces(self, task: str) -> Dict[str, Any]:
        """Summarize trace activity."""
        all_traces = self.env.read(task)
        by_type: Dict[str, int] = defaultdict(int)
        agents_used: Set[str] = set()

        for t in all_traces:
            by_type[t.type] += 1
            agents_used.add(t.agent)

        return {
            "types": dict(by_type),
            "agents": list(agents_used),
            "total": len(all_traces),
        }
