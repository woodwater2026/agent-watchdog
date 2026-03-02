"""
Agent Watchdog + CrewAI integration example.

Addresses CrewAI issue #4495: infinite tool-use loop regression.
Addresses CrewAI issue #4132: token tracking enhancement.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt, HaltReason

watchdog = AgentWatchdog(
    max_budget_usd=1.0,
    max_identical_calls=3,   # catches the infinite loop regression
    timeout_seconds=300,
    model="anthropic/claude-sonnet-4-6",
)

# Option 1: Wrap the entire crew.kickoff() call (simplest)
def run_crew_with_watchdog(crew, inputs: dict):
    """
    Drop-in replacement for crew.kickoff().
    Protects against infinite loops and cost overruns.
    """
    try:
        with watchdog.watch(run_id="crew-run"):
            result = crew.kickoff(inputs=inputs)
            return result
    except WatchdogHalt as e:
        report = e.report
        if report.reason == HaltReason.LOOP_DETECTED:
            print(f"[Watchdog] Infinite loop detected after {len(report.tool_calls)} calls")
            print(f"[Watchdog] Last output: {report.last_output}")
        elif report.reason == HaltReason.BUDGET_EXCEEDED:
            print(f"[Watchdog] Budget exceeded: ${report.estimated_cost_usd:.4f}")
        return {"error": report.reason, "report": report}


# Option 2: Custom CrewAI tool wrapper with per-call tracking
# from crewai.tools import BaseTool
#
# class WatchdogTool(BaseTool):
#     """Wraps any CrewAI tool with watchdog monitoring."""
#     name: str
#     description: str
#     wrapped_tool: BaseTool
#     watchdog: AgentWatchdog
#
#     def _run(self, *args, **kwargs):
#         # Record the call before executing
#         self.watchdog.record_tool_call(
#             self.name,
#             args=str(args) + str(kwargs)
#         )
#         result = self.wrapped_tool._run(*args, **kwargs)
#         self.watchdog.record_tool_call(
#             self.name,
#             args=str(args) + str(kwargs),
#             output=str(result)
#         )
#         return result


# Usage example:
# from crewai import Agent, Task, Crew
#
# researcher = Agent(role="Researcher", goal="...", backstory="...")
# task = Task(description="Research AI trends", agent=researcher)
# crew = Crew(agents=[researcher], tasks=[task])
#
# result = run_crew_with_watchdog(crew, inputs={"topic": "AI agents"})
