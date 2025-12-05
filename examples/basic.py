import asyncio

from collective import Collective, agent


async def example_basic():
    """Simple three-stage pipeline with mock functions."""

    # Define agents using the decorator
    @agent("analyzer", consumes=["goal"], produces="analysis")
    async def analyze(ctx):
        goal = ctx["goal"]["goal"]
        return {"analysis": f"Analyzed: {goal}", "keywords": ["api", "rest", "crud"]}

    @agent("planner", consumes=["analysis"], produces="plan")
    async def plan(ctx):
        analysis = ctx["analysis"]["analysis"]
        return {
            "plan": f"Plan based on: {analysis}",
            "steps": ["design", "implement", "test"],
        }

    @agent("coder", consumes=["plan"], produces="code")
    async def code(ctx):
        steps = ctx["plan"]["steps"]
        return {"code": f"# Implementation for {len(steps)} steps\ndef main(): pass"}

    # Create collective and run
    collective = Collective().add(analyze, plan, code)

    result = await collective.run(
        task="basic_example",
        goal="Build a REST API",
        on_round=lambda r, agents: print(f"Round {r}: {agents}"),
    )

    print(f"\nConverged: {result.converged}")
    print(f"Rounds: {result.rounds}")
    print(f"Code: {collective.result('basic_example', 'code')}")


if __name__ == "__main__":
    asyncio.run(example_basic())
