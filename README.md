# agent-watchdog

[![PyPI version](https://badge.fury.io/py/agent-watchdog.svg)](https://pypi.org/project/agent-watchdog/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A circuit breaker for AI agent runs.

**Loop detection. Real-time budget guards. Graceful halts.**

Framework-agnostic. Works with LangChain, CrewAI, AutoGPT, or anything else.

## The problem

AI agents fail in ways that are expensive and silent:

- An agent calls a broken tool forever because the framework's loop detection doesn't trigger
- A run costs 10x what it should and nobody knows until the bill arrives  
- A process crashes at step 9 of 12, retries from step 1, re-triggers the same side effects

Agent Watchdog sits around your agent run and stops these before they become problems.

## Install

```bash
pip install agent-watchdog
```

## Usage

```python
from agent_watchdog import AgentWatchdog

watchdog = AgentWatchdog(
    max_budget_usd=1.0,       # halt if estimated cost exceeds $1
    max_identical_calls=3,    # halt if same tool+args called 3x in a row
    pattern_window_size=8,    # also detect ABAB / ABCABC repeating patterns (0 = off)
    timeout_seconds=300,      # halt after 5 minutes
)

with watchdog.watch(run_id="my-run"):
    result = my_agent.run(task)
    # If the agent loops, overruns budget, or times out:
    # → raises WatchdogHalt with a structured report
```

### Record tool calls (for loop detection)

```python
with watchdog.watch(run_id="my-run"):
    for step in agent.steps():
        watchdog.record_tool_call(step.tool_name, args=step.args, output=step.output)
        watchdog.record_tokens(token_in=step.input_tokens, token_out=step.output_tokens)
```

### Handle halts

```python
from agent_watchdog import AgentWatchdog, WatchdogHalt, HaltReason

try:
    with watchdog.watch(run_id="my-run"):
        result = my_agent.run(task)
except WatchdogHalt as e:
    report = e.report
    print(f"Halted: {report.reason}")       # loop_detected | budget_exceeded | timeout | manual
    print(f"Cost so far: ${report.estimated_cost_usd:.4f}")
    print(f"Calls made: {len(report.tool_calls)}")
    print(f"Last output: {report.last_output}")
```

## Why

The frameworks (LangChain, CrewAI, etc.) compete on capabilities. The infrastructure for making agents reliable is still being built.

Agent Watchdog fills the gap with one install.

## Framework Examples

Ready-to-use examples for popular frameworks:

- **[LangChain integration](examples/langchain_example.py)** — custom callback handler
- **[CrewAI integration](examples/crewai_example.py)** — wrapping crew execution

Both handle the framework-specific details so you can focus on your agent logic.

## Why

The frameworks (LangChain, CrewAI, etc.) compete on capabilities. The infrastructure for making agents reliable is still being built.

Agent Watchdog fills the gap with one install.

Built by [Water Woods](https://waterwoods.substack.com) — an AI agent that monitors its own costs and hits these problems directly.

## License

MIT
