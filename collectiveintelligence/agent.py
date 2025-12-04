from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable, Set

from .environment import Environment


@dataclass
class Agent:
    """
    A specialized agent that acts when conditions are met.

    Args:
        name: Unique identifier
        consumes: List of trace types this agent needs to activate
        produces: Trace type this agent produces
        execute: Async function (Dict) -> Dict
        run_once: If True, agent only fires once per task
    """

    name: str
    consumes: List[str]
    produces: str
    execute: Callable[[Dict[str, Any]], Any]
    run_once: bool = False
    _has_run: Set[str] = field(default_factory=set)

    def can_activate(self, env: Environment, task: str) -> bool:
        """
        Check if agent should act based on trace availability.

        Returns True if:
        1. Agent hasn't already run (if run_once=True)
        2. All required input traces exist and are alive
        3. No alive output exists (invalidated outputs don't count)
        """
        if self.run_once and task in self._has_run:
            return False

        for trace_type in self.consumes:
            if not env.read(task, trace_type):
                return False

        existing = env.strongest(task, self.produces)
        return existing is None or existing.invalidated

    def build_context(self, env: Environment, task: str) -> Dict[str, Any]:
        """Build context dict from consumed traces."""
        context = {}
        for trace_type in self.consumes:
            traces = env.read(task, trace_type)
            if traces:
                strongest = max(traces, key=lambda t: t.strength)
                context[trace_type] = strongest.data
        return context

    async def act(self, env: Environment, task: str) -> Dict[str, Any]:
        """Execute agent's function and leave trace."""
        context = self.build_context(env, task)

        result = await self.execute(context)

        # Normalize result to dict
        if not isinstance(result, dict):
            result = {"result": result}

        await env.add(agent=self.name, task=task, type=self.produces, data=result)

        if self.run_once:
            self._has_run.add(task)

        return result


def agent(
    name: str, consumes: List[str], produces: str, run_once: bool = False
) -> Callable[[Callable], Agent]:
    """
    Decorator to create agents from async functions.

    Usage:
        @agent("analyzer", consumes=["goal"], produces="analysis")
        async def analyze(ctx):
            return {"analysis": "..."}

        @agent("init", consumes=["goal"], produces="config", run_once=True)
        async def initialize(ctx):
            return {"config": "..."}
    """

    def decorator(fn: Callable[[Dict], Any]) -> Agent:
        return Agent(
            name=name,
            consumes=consumes,
            produces=produces,
            execute=fn,
            run_once=run_once,
        )

    return decorator
