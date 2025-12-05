# Collective Intelligence

Minimal swarm intelligence library for LLM orchestration. Agents communicate through stigmergy (indirect coordination via shared environment) rather than direct messaging.

## Quick Start

```python
import asyncio
from collective import Collective, agent

@agent("analyzer", consumes=["goal"], produces="analysis")
async def analyze(ctx):
    goal = ctx["goal"]["goal"]
    return {"analysis": f"Analyzed: {goal}"}

@agent("coder", consumes=["analysis"], produces="code")
async def code(ctx):
    analysis = ctx["analysis"]["analysis"]
    return {"code": f"# Based on: {analysis}\ndef main(): pass"}

async def main():
    collective = Collective().add(analyze, code)
    result = await collective.run(task="demo", goal="Build an API")

    print(f"Converged: {result.converged}")
    print(f"Code: {collective.result('demo', 'code')}")

asyncio.run(main())
```

## API

### Define Agents

```python
# Using decorator
@agent("name", consumes=["input_type"], produces="output_type")
async def my_agent(ctx):
    data = ctx["input_type"]  # Dict from strongest trace
    return {"result": "..."}  # Becomes trace data

# One-shot (runs once per task)
@agent("init", consumes=["goal"], produces="config", run_once=True)
async def initialize(ctx):
    return {"model": "gpt-5"}
```

### Run Collective

```python
collective = Collective(
    decay_rates={"ephemeral": 0.5}  # Optional: custom decay per type
)
collective.add(agent1, agent2, agent3)

result = await collective.run(
    task="unique_task_id",
    goal="What to accomplish",
    max_rounds=10,
    on_round=lambda r, agents: print(f"Round {r}: {agents}")
)

# Result
result.converged      # bool
result.rounds         # int
result.cycle_detected # bool
result.errors         # List[Dict]
```

### Get Results

```python
collective.result("task", "code")           # Single type
collective.results("task", "code", "plan")  # Multiple types
```

### Environment Operations

```python
env = collective.env

env.strongest("task", "type")      # Get strongest trace
env.freshest("task", "type")       # Get most recent trace
env.strongest_data("task", "type") # Get data directly

env.invalidate("task", type="code")  # Trigger re-computation
env.reinforce("task", "type", 0.3)   # Strengthen traces
env.has_errors("task")               # Check for errors
env.observe("task")                  # Debug print all traces
```

## Patterns

### Parallel Agents

Agents consuming the same input run in parallel:

```python
@agent("researcher", consumes=["goal"], produces="research")
async def research(ctx): ...

@agent("competitor_analyst", consumes=["goal"], produces="competitors")
async def analyze(ctx): ...

@agent("strategist", consumes=["research", "competitors"], produces="strategy")
async def strategize(ctx): ...

# Round 1: researcher + competitor_analyst (parallel)
# Round 2: strategist (waits for both)
```

### Validation Loop

```python
@agent("generator", consumes=["goal"], produces="output")
async def generate(ctx): ...

@agent("validator", consumes=["output"], produces="validation")
async def validate(ctx):
    if not good_enough(ctx["output"]):
        return {"valid": False}
    return {"valid": True}

# After run, check and retry:
if not collective.result("task", "validation")["valid"]:
    collective.env.invalidate("task", type="output")
    await collective.run(task="task", goal="...")
```

### With Real LLMs

```python
import os
import asyncio
from anthropic import AsyncAnthropic

client = anthropic.AsyncAnthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

async def call_claude(prompt: str) -> str:
    message = await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="claude-sonnet-4-5-20250929",
    )
    return message.content

@agent("writer", consumes=["goal"], produces="draft")
async def write(ctx):
    goal = ctx["goal"]["goal"]
    text = await call_claude(f"Write about: {goal}")
    return {"draft": text}
```
