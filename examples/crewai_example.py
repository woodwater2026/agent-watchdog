"""
Agent Watchdog + CrewAI integration example.

Addresses the real issue: CrewAI agents entering infinite tool-call loops.
(See: https://github.com/crewAIInc/crewAI/issues/4495)
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt

# --- Setup (replace with your actual CrewAI crew) ---
# from crewai import Agent, Task, Crew
# crew = Crew(agents=[...], tasks=[...])

watchdog = AgentWatchdog(
    max_budget_usd=1.0,
    max_identical_calls=3,    # the exact failure mode from CrewAI issue #4495
    timeout_seconds=300,
    model="anthropic/claude-sonnet-4-6",
)

def run_crew_safely(inputs: dict, run_id: str = "crewai-run"):
    """Run a CrewAI crew with watchdog protection."""
    try:
        with watchdog.watch(run_id=run_id):
            result = crew.kickoff(inputs=inputs)
            return result

    except WatchdogHalt as e:
        r = e.report
        print(f"\n[Watchdog stopped the crew]")
        print(f"  Reason: {r.reason.value}")
        print(f"  After: {r.elapsed_seconds:.0f}s, {len(r.tool_calls)} tool calls")
        print(f"  Cost: ${r.estimated_cost_usd:.4f}")
        if r.reason.value == "loop_detected":
            print(f"  → Identical tool calls detected. Check your tool wrappers.")
        return {"error": r.reason.value, "report": r}


# --- CrewAI callback (optional, for loop detection) ---
# CrewAI doesn't have a standard callback interface yet.
# Workaround: patch tool execution at the BaseTool level.

# from crewai.tools import BaseTool
# _original_run = BaseTool._run
#
# def _patched_run(self, *args, **kwargs):
#     watchdog.record_tool_call(self.name, args=str(args))
#     return _original_run(self, *args, **kwargs)
#
# BaseTool._run = _patched_run
