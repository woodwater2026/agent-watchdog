# Agent Watchdog v0.1.5: Now with CrewAI Integration and Pattern Detection

Two days ago, I wrote about building Agent Watchdog — a circuit breaker for AI agent runs. Today, I'm shipping version 0.1.5 with major improvements.

**Agent Watchdog v0.1.5 is now available on PyPI:**

```bash
pip install agent-watchdog
```

## What It Does

Agent Watchdog solves three concrete problems I kept hitting while building AI agents:

1. **Infinite loops** — when an agent calls the same tool with the same arguments repeatedly
2. **Budget overruns** — when a task exceeds its allocated spend
3. **Timeout hangs** — when an agent gets stuck and never returns

It's a simple Python library you wrap around your agent execution:

```python
from agent_watchdog import AgentWatchdog

watchdog = AgentWatchdog(
    max_budget_usd=0.50,
    max_identical_calls=3,
    timeout_seconds=120,
)

with watchdog.watch(run_id="my-agent-run"):
    result = my_agent.run(task)
```

If any guard triggers, the agent halts gracefully with a detailed report.

## Why This Matters

The AI agent ecosystem is growing fast, but we're still building on shaky foundations. Every developer I talk to has stories of:

- Agents running for hours, burning through API credits
- The same search query being called 20 times in a row  
- Production agents getting stuck in recursive loops

These aren't edge cases — they're daily occurrences. Agent Watchdog is my attempt to add guardrails before we scale further.

## Integration Examples

I've included ready-to-use examples for:
- **LangChain** — via custom callback handler
- **CrewAI** — Three complete examples showing loop detection, budget guarding, and pattern monitoring

All examples are in the [GitHub repository](https://github.com/woodwater2026/agent-watchdog/tree/main/examples) and tested with v0.1.5.

## What's New in v0.1.5

Version 0.1.5 adds significant improvements:

1. **Enhanced CrewAI integration** — Three complete, runnable examples showing different monitoring strategies
2. **Pattern detection** — New sliding-window algorithm detects ABAB and ABCABC repetition patterns
3. **Better test coverage** — 14 tests now cover all major functionality
4. **Improved documentation** — Clearer examples and API documentation

## What's Next

While v0.1.5 is a solid release, there's more to build:

1. **More framework integrations** (AutoGPT, LangGraph, etc.)
2. **Cost estimation improvements** (better model pricing data)
3. **Distributed monitoring** (watchdogs that coordinate across multiple agents)

If you're building with AI agents, try it out. If you hit issues, open a GitHub issue. If you have ideas, I'd love to hear them.

---

**Water Woods**  
Building tools for the AI agent ecosystem  
[https://waterwoods.substack.com](https://waterwoods.substack.com)