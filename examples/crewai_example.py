"""
Agent Watchdog + CrewAI integration example.

CrewAI doesn't expose per-step callbacks as easily as LangChain,
but you can wrap the crew.kickoff() call for budget + timeout protection.
For loop detection, patch individual tool wrappers.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt


def run_crew_with_watchdog(crew, inputs: dict, run_id: str = "crew-run"):
    """
    Run a CrewAI crew with timeout and budget protection.

    Loop detection requires instrumenting individual tools (see WatchdogTool below).

    Args:
        crew: Your configured CrewAI Crew instance
        inputs: Dict of inputs for the crew
        run_id: Identifier for logging

    Returns:
        dict with 'output' on success, 'halted' and 'report' on WatchdogHalt
    """
    watchdog = AgentWatchdog(
        max_budget_usd=2.0,
        max_identical_calls=3,
        timeout_seconds=300,
    )

    try:
        with watchdog.watch(run_id=run_id):
            result = crew.kickoff(inputs=inputs)
        return {"output": result, "halted": False}

    except WatchdogHalt as e:
        return {
            "halted": True,
            "report": e.report,
            "partial_output": e.report.last_output,
        }


class WatchdogTool:
    """
    Wrap a CrewAI BaseTool to feed call data into a watchdog.

    Usage:
        from crewai.tools import BaseTool
        from agent_watchdog import AgentWatchdog

        watchdog = AgentWatchdog(max_identical_calls=3)

        class MySearchTool(BaseTool):
            name = "web_search"
            description = "Search the web"

            def _run(self, query: str) -> str:
                watchdog.record_tool_call("web_search", args=query)
                # ... actual search logic
                result = do_search(query)
                return result
    """
    pass


# --- Minimal runnable demo ---
if __name__ == "__main__":
    try:
        from crewai import Agent, Task, Crew
        from crewai.tools import BaseTool

        watchdog = AgentWatchdog(
            max_budget_usd=0.50,
            max_identical_calls=3,
            timeout_seconds=60,
        )

        class BrokenTool(BaseTool):
            name: str = "broken_tool"
            description: str = "A tool that always fails"

            def _run(self, query: str) -> str:
                watchdog.record_tool_call(self.name, args=query)
                return "Error: tool unavailable"

        broken = BrokenTool()

        researcher = Agent(
            role="Researcher",
            goal="Find information about AI agents",
            backstory="You are a researcher.",
            tools=[broken],
            verbose=False,
        )

        task = Task(
            description="Research the latest developments in AI agent frameworks",
            expected_output="A summary of findings",
            agent=researcher,
        )

        crew = Crew(agents=[researcher], tasks=[task], verbose=False)

        print("Running crew with watchdog (budget=$0.50, timeout=60s)...")
        with watchdog.watch(run_id="demo-crew"):
            try:
                result = run_crew_with_watchdog(crew, {}, run_id="demo-crew")
            except WatchdogHalt as e:
                r = e.report
                print(f"✓ Watchdog halted crew: {r.reason.value}")
                print(f"  Cost: ${r.estimated_cost_usd:.4f}")

    except ImportError as e:
        print(f"Demo requires crewai: {e}")
        print("Install with: pip install crewai")
