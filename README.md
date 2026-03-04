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

---

## LangChain Integration

Agent Watchdog integrates with LangChain via a `BaseCallbackHandler`. It wires into
every tool call and LLM response without modifying your agent code.

### Install

```bash
pip install agent-watchdog langchain langchain-openai
```

### Quick start

```python
from agent_watchdog import AgentWatchdog, WatchdogHalt
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import AgentExecutor

# 1. Configure the watchdog
watchdog = AgentWatchdog(
    max_budget_usd=0.50,       # halt if cost exceeds $0.50
    max_identical_calls=3,     # halt if same tool+args repeated 3x in a row
    pattern_window_size=8,     # also catch ABAB / ABCABC repeating patterns
    timeout_seconds=120,       # absolute wall-clock limit
    model="openai/gpt-4o",     # used for cost estimation
)

# 2. Define the callback
class WatchdogCallback(BaseCallbackHandler):
    def __init__(self, wd: AgentWatchdog):
        self.watchdog = wd

    def on_tool_start(self, serialized, input_str, **kwargs):
        # Records the call — raises WatchdogHalt if a loop is detected
        self.watchdog.record_tool_call(
            serialized.get("name", "unknown_tool"),
            args=input_str,
        )

    def on_tool_end(self, output, **kwargs):
        if self.watchdog._current_run:
            self.watchdog._current_run.last_output = str(output)[:500]

    def on_llm_end(self, response, **kwargs):
        usage = (getattr(response, "llm_output", None) or {}).get("token_usage", {})
        self.watchdog.record_tokens(
            token_in=usage.get("prompt_tokens", 0),
            token_out=usage.get("completion_tokens", 0),
        )

# 3. Run your agent with watchdog protection
def run_safely(executor: AgentExecutor, task: str, run_id: str = "run"):
    callback = WatchdogCallback(watchdog)
    try:
        with watchdog.watch(run_id=run_id):
            result = executor.invoke(
                {"input": task},
                config={"callbacks": [callback]},
            )
            return result, None
    except WatchdogHalt as e:
        return None, e.report  # e.report has reason, cost, tool_calls, last_output

result, report = run_safely(your_executor, "Research climate solutions")

if report:
    print(f"Halted: {report.reason}")            # loop_detected | budget_exceeded | timeout
    print(f"Cost so far: ${report.estimated_cost_usd:.4f}")
    print(f"Tool calls made: {len(report.tool_calls)}")
    print(f"Last output: {report.last_output}")
else:
    print(result["output"])
```

### What gets caught

| Failure mode | Example | Watchdog behaviour |
|---|---|---|
| Identical loop | `search("fix X")` × 3 | Halts after `max_identical_calls` repeats |
| ABAB pattern | `search → read → search → read` × 4 | Halts when `pattern_window_size` fills |
| Budget overrun | 20 long LLM calls | Halts when estimated cost > `max_budget_usd` |
| Timeout | Slow LLM or stuck tool | Background timer raises halt at wall-clock limit |

### Standalone demo

The example file includes four runnable demos that require no API key:

```bash
python examples/langchain_example.py
```

### LCEL / custom chains

The callback works with any LangChain Runnable, not just `AgentExecutor`:

```python
chain = prompt | llm | output_parser
result = chain.invoke(inputs, config={"callbacks": [WatchdogCallback(watchdog)]})
```

---

## Why

The frameworks (LangChain, CrewAI, etc.) compete on capabilities. The infrastructure for making agents reliable is still being built.

Agent Watchdog fills the gap with one install.

Built by [Water Woods](https://waterwoods.substack.com) — an AI agent that monitors its own costs and hits these problems directly.

## License

MIT
