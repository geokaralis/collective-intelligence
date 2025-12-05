import pytest
from collective import Collective, agent, Environment


class TestBasicPipeline:
    """Test basic agent pipeline execution."""

    @pytest.mark.asyncio
    async def test_linear_pipeline_converges(self):
        """Three agents in sequence should converge."""

        @agent("analyzer", consumes=["goal"], produces="analysis")
        async def analyze(ctx):
            goal = ctx["goal"]["goal"]
            return {"analysis": f"Analyzed: {goal}"}

        @agent("planner", consumes=["analysis"], produces="plan")
        async def plan(ctx):
            analysis = ctx["analysis"]["analysis"]
            return {"plan": f"Plan for: {analysis}", "steps": ["a", "b", "c"]}

        @agent("coder", consumes=["plan"], produces="code")
        async def code(ctx):
            steps = ctx["plan"]["steps"]
            return {"code": f"def main(): pass  # {len(steps)} steps"}

        collective = Collective().add(analyze, plan, code)
        result = await collective.run(task="test_linear", goal="Build an API")

        assert result.converged
        assert result.rounds <= 5
        assert "analyzer" in result.traces["agents"]
        assert "planner" in result.traces["agents"]
        assert "coder" in result.traces["agents"]

        code_result = collective.result("test_linear", "code")
        assert code_result is not None
        assert "def main" in code_result["code"]

    @pytest.mark.asyncio
    async def test_parallel_agents_run_together(self):
        """Agents consuming the same input should run in parallel."""
        execution_order = []

        @agent("agent_a", consumes=["goal"], produces="output_a")
        async def agent_a(ctx):
            execution_order.append("a")
            return {"from": "a"}

        @agent("agent_b", consumes=["goal"], produces="output_b")
        async def agent_b(ctx):
            execution_order.append("b")
            return {"from": "b"}

        @agent("combiner", consumes=["output_a", "output_b"], produces="combined")
        async def combine(ctx):
            execution_order.append("combiner")
            return {"a": ctx["output_a"], "b": ctx["output_b"]}

        collective = Collective().add(agent_a, agent_b, combine)
        result = await collective.run(task="test_parallel", goal="Test")

        assert result.converged
        # a and b should run before combiner
        assert execution_order.index("combiner") > execution_order.index("a")
        assert execution_order.index("combiner") > execution_order.index("b")

        combined = collective.result("test_parallel", "combined")
        assert combined is not None


class TestEnvironment:
    """Test environment trace operations."""

    def test_strongest_returns_highest_strength(self):
        """strongest() should return trace with highest strength."""
        env = Environment()

        env.add_sync("agent1", "task", "data", {"version": 1})
        env.add_sync("agent2", "task", "data", {"version": 2})

        # Manually set strengths
        env._traces["task"][0].strength = 0.5
        env._traces["task"][1].strength = 0.9

        strongest = env.strongest("task", "data")
        assert strongest is not None
        assert strongest.data["version"] == 2

    def test_invalidate_marks_traces(self):
        """invalidate() should mark traces as invalid."""
        env = Environment()

        env.add_sync("agent1", "task", "output", {"value": 1})
        trace = env.strongest("task", "output")
        assert trace is not None
        assert trace.alive

        env.invalidate("task", type="output")

        trace = env.strongest("task", "output")
        assert trace is None  # No alive traces

    def test_decay_weakens_traces(self):
        """decay() should reduce trace strength."""
        env = Environment()

        env.add_sync("agent1", "task", "data", {"value": 1})
        initial_strength = env._traces["task"][0].strength

        env.decay("task", rates={"data": 0.5})

        final_strength = env._traces["task"][0].strength
        assert final_strength < initial_strength
        assert final_strength == pytest.approx(0.5)

    def test_observe_returns_debug_string(self):
        """observe() should return readable debug output."""
        env = Environment()

        env.add_sync("analyzer", "task", "analysis", {"result": "test"})

        output = env.observe("task")
        assert "analyzer" in output
        assert "analysis" in output
        assert "str=" in output
