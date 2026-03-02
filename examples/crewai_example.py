"""
Agent Watchdog + CrewAI integration example.

CrewAI doesn't have a standard callback system like LangChain,
so we wrap at the Crew.kickoff() level and use a custom BaseTool wrapper.

Requirements:
    pip install agent-watchdog crewai
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt
from crewai.tools import BaseTool
from typing import Any


class WatchdogTool(BaseTool):
    """
    Wraps any CrewAI BaseTool with watchdog monitoring.
    Automatically records tool calls for loop detection.
    """
    name: str = "watchdog_tool"
    description: str = ""
    _inner: Any
    _watchdog: Any

    def __init__(self, tool: BaseTool, watchdog: AgentWatchdog):
        super().__init__(
            name=tool.name,
            description=tool.description,
        )
        object.__setattr__(self, '_inner', tool)
        object.__setattr__(self, '_watchdog', watchdog)

    def _run(self, *args, **kwargs) -> str:
        result = self._inner._run(*args, **kwargs)
        self._watchdog.record_tool_call(self.name, args=(args, kwargs), output=result)
        return result


def run_crew_with_watchdog(crew, run_id: str = "crew-run", max_budget_usd: float = 1.0):
    """
    Run a CrewAI crew under watchdog protection.

    Usage:
        result, report = run_crew_with_watchdog(my_crew, run_id="research-task")
    """
    watchdog = AgentWatchdog(
        max_budget_usd=max_budget_usd,
        max_identical_calls=3,
        timeout_seconds=300,
    )

    # Wrap all agent tools with WatchdogTool
    for agent in crew.agents:
        agent.tools = [WatchdogTool(t, watchdog) for t in agent.tools]

    try:
        with watchdog.watch(run_id=run_id):
            result = crew.kickoff()
        return result, None

    except WatchdogHalt as e:
        return None, e.report


# --- Demo ---
if __name__ == "__main__":
    from crewai import Agent, Task, Crew
    from crewai.tools import BaseTool

    call_count = 0

    class BrokenSearchTool(BaseTool):
        name: str = "web_search"
        description: str = "Search the web"

        def _run(self, query: str) -> str:
            global call_count
            call_count += 1
            return "No results. Try refining your query."

    researcher = Agent(
        role="Researcher",
        goal="Find information about AI agent frameworks",
        backstory="You are a meticulous researcher.",
        tools=[BrokenSearchTool()],
        verbose=True,
    )

    task = Task(
        description="Research the latest developments in AI agent frameworks",
        expected_output="A summary of current AI agent frameworks",
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task])

    print("Running crew with a broken tool...")
    result, report = run_crew_with_watchdog(crew, run_id="demo-crew")

    if report:
        print(f"\nWatchdog caught: {report.reason.value}")
        print(f"Calls made: {len(report.tool_calls)}")
    else:
        print(f"\nCrew result: {result}")
