"""
Agent Watchdog + CrewAI integration example.

Wraps a CrewAI crew with watchdog monitoring.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt

# pip install agent-watchdog crewai


watchdog = AgentWatchdog(
    max_budget_usd=1.0,
    max_identical_calls=3,
    timeout_seconds=120,
    model="anthropic/claude-sonnet-4-6",
)


def run_crew_with_watchdog(crew, inputs: dict, run_id: str = "crew-run") -> dict:
    """
    Run a CrewAI crew with watchdog monitoring.

    CrewAI doesn't expose callbacks as cleanly as LangChain,
    so this uses the basic context manager approach.
    For per-tool monitoring, patch your custom BaseTool subclasses
    to call watchdog.record_tool_call() in their _run() method.
    """
    try:
        with watchdog.watch(run_id=run_id):
            result = crew.kickoff(inputs=inputs)
            return {"ok": True, "output": str(result)}

    except WatchdogHalt as e:
        report = e.report
        return {
            "ok": False,
            "halted": True,
            "reason": report.reason.value,
            "cost_usd": report.estimated_cost_usd,
            "message": report.message,
        }


# --- Per-tool integration ---

from crewai.tools import BaseTool
from typing import Any


class WatchedTool(BaseTool):
    """
    Base class for CrewAI tools that auto-report to AgentWatchdog.
    Subclass this instead of BaseTool to get loop detection for free.
    """
    watchdog: Any = None

    def _run(self, *args, **kwargs):
        if self.watchdog:
            self.watchdog.record_tool_call(self.name, args=(args, kwargs))
        return self._watched_run(*args, **kwargs)

    def _watched_run(self, *args, **kwargs):
        raise NotImplementedError


# Example tool using WatchedTool
class SearchTool(WatchedTool):
    name: str = "search"
    description: str = "Search the web."
    watchdog: Any = watchdog

    def _watched_run(self, query: str) -> str:
        return f"Results for: {query}"


# --- Usage example ---
# from crewai import Agent, Task, Crew
#
# search_tool = SearchTool()
#
# researcher = Agent(
#     role="Researcher",
#     goal="Find relevant information",
#     tools=[search_tool],
#     llm="anthropic/claude-haiku-4-5",
# )
#
# task = Task(
#     description="Research the latest developments in AI agent frameworks.",
#     agent=researcher,
#     expected_output="A summary of findings.",
# )
#
# crew = Crew(agents=[researcher], tasks=[task])
# result = run_crew_with_watchdog(crew, inputs={}, run_id="research-001")
# print(result)
