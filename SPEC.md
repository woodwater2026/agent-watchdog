# Agent Watchdog — SPEC

**Status**: Planning  
**Started**: 2026-03-01  
**Author**: Water Woods

---

## Problem

AI agents fail in ways that are expensive and silent.

Three specific failure modes I've observed directly and in the wild:

### 1. Infinite loops
An agent calls the same tool repeatedly — same args, no progress — because the framework's iteration limit is either not triggered, not enforced, or fails silently. The agent doesn't know it's stuck. The developer doesn't know either until the bill arrives or a timeout kills the process.

First-hand: heartbeat storm today — multiple heartbeat triggers queued while the agent was mid-task, stacking up, each one potentially triggering another turn. No built-in detection.

In production: CrewAI issue #4495 (23 comments) — tool wrapper regression causes infinite tool-call loop. Agents spend money calling a broken tool forever.

### 2. Runaway cost
No real-time signal when a run is going wrong. Developer sees cost after the fact — end of the run, end of the day, end of the month. An agent running 100x/day at $0.30/run = $900/month, silently.

### 3. No graceful halt
When something does stop an agent (iteration limit, timeout, exception), the default behavior is: crash or return nothing. No summary of what was accomplished. No signal to the caller. No record of what happened.

---

## What Agent Watchdog Does

A lightweight, framework-agnostic process monitor that wraps agent runs and enforces three things:

### 1. Loop Detection
- Track tool call history: (tool_name, args_hash) per run
- If same (tool, args) pair appears N times with no different calls in between → loop detected
- Configurable threshold (default: 3 identical consecutive calls)
- On detect: halt the agent, emit a LOOP_DETECTED event

### 2. Cost Guard
- Real-time token counting per run (integrates with agent-budget-guard)
- Configurable per-run budget cap (default: $1.00)
- On exceed: halt with BUDGET_EXCEEDED, log cost so far
- Warning at 80% threshold

### 3. Graceful Halt + Summary
- On any halt (loop, budget, timeout, manual): collect what was accomplished
- Emit structured halt report: reason, tools called, cost incurred, last output
- No silent crashes

---

## Minimum Viable Version (v0.1)

Single Python class: `AgentWatchdog`

```python
from agent_watchdog import AgentWatchdog

watchdog = AgentWatchdog(
    max_budget_usd=1.0,
    max_identical_calls=3,
    timeout_seconds=300,
)

with watchdog.watch(run_id="my-run-001"):
    result = my_agent.run("do the thing")

# If agent loops, exceeds budget, or times out:
# → watchdog halts it and raises WatchdogHalt with a report
```

### What v0.1 ships:
- `AgentWatchdog` class with context manager interface
- Loop detection (call history tracking)
- Budget cap (token counting via agent-budget-guard)
- Timeout enforcement
- `WatchdogHalt` exception with structured report
- CLI: `watchdog watch --budget 1.0 --timeout 300 python my_agent.py`
- Zero framework dependencies (works with LangChain, CrewAI, anything)

### What v0.1 does NOT ship:
- Dashboard / UI
- Multi-agent coordination monitoring
- Cloud logging
- Framework-native integrations (LangChain callback, CrewAI hook)

Those are v0.2+.

---

## Integration with agent-budget-guard

agent-budget-guard tracks cost per task label across sessions.  
Agent Watchdog tracks cost per run in real time.

They're complementary:
- agent-budget-guard = daily ledger (did I stay under budget today?)
- agent-watchdog = per-run circuit breaker (is this specific run going wrong right now?)

In code: Watchdog calls `agent-budget-guard` APIs to log halt events as tasks.  
Budget guard reports include "halted runs" as a category.

---

## Distribution

- PyPI: `pip install agent-watchdog`
- GitHub: github.com/woodwater2026/agent-watchdog
- Works standalone, integrates with agent-budget-guard if installed

---

## Why This, Why Now

The frameworks (LangChain, CrewAI) are competing on capabilities.  
Nobody is competing on reliability infrastructure.  
Every developer building production agents is writing their own version of this — ad-hoc timeouts, manual loop guards, custom budget checks.

That's the gap. agent-watchdog fills it with one install.
