"""
Agent Watchdog + CrewAI integration example.

Addresses the CrewAI issue #4495: tool wrapper regression causing infinite loops.
With watchdog, the loop is caught after N identical calls and halted gracefully.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt, HaltReason

# --- CrewAI setup (simplified) ---
# from crewai import Agent, Task, Crew
# from crewai.tools import BaseTool

# class MyTool(BaseTool):
#     name: str = "search"
#     description: str = "Search the web"
#
#     def _run(self, query: str) -> str:
#         return f"Results for: {query}"

# researcher = Agent(
#     role="Researcher",
#     goal="Research the topic thoroughly",
#     tools=[MyTool()],
# )

# task = Task(
#     description="Research AI agent reliability",
#     agent=researcher,
# )

# crew = Crew(agents=[researcher], tasks=[task])

# --- Watchdog integration ---

watchdog = AgentWatchdog(
    max_budget_usd=1.0,
    max_identical_calls=3,   # catches the infinite loop regression in CrewAI 1.9.3+
    timeout_seconds=300,
    model="anthropic/claude-sonnet-4-6",
)


def run_crew_with_watchdog(crew, run_id: str = "crewai-run"):
    """
    Run a CrewAI crew with watchdog protection.

    Specifically addresses: CrewAI issue #4495 where tool wrappers
    enter infinite loops calling _run() with no args.
    """
    try:
        with watchdog.watch(run_id=run_id):
            # Hook into CrewAI's step callback to record tool calls
            # crew.step_callback = lambda step: _record_step(step)
            # result = crew.kickoff()
            result = "crew result"  # placeholder
            return result

    except WatchdogHalt as e:
        report = e.report
        print(f"[Watchdog] Crew halted: {report.reason.value}")
        print(f"  Calls made: {len(report.tool_calls)}")
        print(f"  Cost incurred: ${report.estimated_cost_usd:.4f}")
        print(f"  Time elapsed: {report.elapsed_seconds:.1f}s")

        if report.reason == HaltReason.LOOP_DETECTED:
            # This is the CrewAI #4495 scenario
            print(f"  Loop detected: {report.message}")
            print("  Check your tool wrapper for the regression.")

        return None


def _record_step(step):
    """Hook for CrewAI step callback."""
    if hasattr(step, "tool") and step.tool:
        watchdog.record_tool_call(
            tool_name=step.tool,
            args=getattr(step, "tool_input", None),
            output=getattr(step, "result", None),
        )


if __name__ == "__main__":
    # run_crew_with_watchdog(crew, run_id="research-001")
    print("CrewAI integration example — see comments for usage")
