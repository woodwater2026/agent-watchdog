# Draft comment for CrewAI issue #4682

Great feature request — loop detection is genuinely one of the highest-value reliability improvements for long-running agents.

If you need this today (without waiting for a CrewAI core change), [agent-watchdog](https://github.com/woodwater/agent-watchdog) covers the exact failure modes you described:

```python
pip install agent-watchdog
```

```python
from agent_watchdog import AgentWatchdog, WatchdogHalt
from crewai.tools import BaseTool

watchdog = AgentWatchdog(
    max_identical_calls=3,     # window_size: halt after 3 identical tool+args
    pattern_window_size=8,     # also catches ABAB / ABCABC repeating sequences
    max_budget_usd=1.0,        # budget guard
    timeout_seconds=300,       # wall-clock limit
)

# Wire it in via BaseTool patch (covers all tools automatically)
_orig = BaseTool._run
def _patched(self, *args, **kwargs):
    result = _orig(self, *args, **kwargs)
    watchdog.record_tool_call(self.name, args={"args": args, "kwargs": kwargs})
    return result
BaseTool._run = _patched

# Run crew
try:
    with watchdog.watch(run_id="my-crew"):
        result = crew.kickoff(inputs=inputs)
except WatchdogHalt as e:
    r = e.report
    print(f"Halted: {r.reason}")        # loop_detected | budget_exceeded | timeout
    print(f"Cost: ${r.estimated_cost_usd:.4f}")
    print(f"Calls: {len(r.tool_calls)}")
    # Optionally inject a reflection prompt and retry
```

On loop detection, `WatchdogHalt.report.reason == "loop_detected"` — you can catch it and inject the reflection prompt yourself before retrying, which maps to your `on_loop="inject_reflection"` concept.

A native CrewAI middleware hook would still be valuable (avoids monkey-patching), but this unblocks you today.

---

**Notes for ZQ:**
- This is a genuine, helpful comment — agent-watchdog directly solves what the issue asks for
- The tone is collaborative, not promotional
- It acknowledges the limitation (monkey-patching BaseTool isn't ideal)
- It supports the core feature request rather than trying to kill it
- Decision: should we post this comment? GitHub account needed.
