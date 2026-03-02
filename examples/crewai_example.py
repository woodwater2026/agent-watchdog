"""
Agent Watchdog + CrewAI integration example.

CrewAI doesn't have a callback system like LangChain, so we wrap
the Crew.kickoff() call and instrument individual tools.

Requires: pip install agent-watchdog crewai
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt
from functools import wraps


def watchdog_tool(watchdog: AgentWatchdog, tool_name: str = None):
    """
    Decorator to wrap a CrewAI tool function with watchdog monitoring.
    Apply to any @tool-decorated function.

    Example:
        @tool("Search the web")
        @watchdog_tool(my_watchdog, tool_name="web_search")
        def search(query: str) -> str:
            return brave_search(query)
    """
    def decorator(fn):
        name = tool_name or fn.__name__
        @wraps(fn)
        def wrapper(*args, **kwargs):
            watchdog.record_tool_call(name, args=(args, kwargs))
            result = fn(*args, **kwargs)
            if watchdog._current_run:
                watchdog._current_run.last_output = str(result)[:500]
            return result
        return wrapper
    return decorator


def run_crew_with_watchdog(crew, run_id: str = "crewai-run", max_budget_usd: float = 1.0):
    """
    Run a CrewAI Crew with watchdog protection.

    Example:
        crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
        result = run_crew_with_watchdog(crew, run_id="my-crew", max_budget_usd=2.0)
    """
    watchdog = AgentWatchdog(
        max_budget_usd=max_budget_usd,
        max_identical_calls=3,
        timeout_seconds=300,
    )

    try:
        with watchdog.watch(run_id=run_id):
            result = crew.kickoff()
        return result
    except WatchdogHalt as e:
        report = e.report
        print(f"\n[Watchdog] Crew halted: {report.reason.value}")
        print(f"  Cost: ${report.estimated_cost_usd:.4f}")
        print(f"  Tool calls recorded: {len(report.tool_calls)}")
        print(f"  Time: {report.elapsed_seconds:.1f}s")
        raise


if __name__ == "__main__":
    print("Agent Watchdog + CrewAI integration")
    print()
    print("Two patterns:")
    print()
    print("1. Wrap tools with @watchdog_tool to record each call:")
    print("   @tool('Search')")
    print("   @watchdog_tool(watchdog, tool_name='search')")
    print("   def search(query: str) -> str: ...")
    print()
    print("2. Wrap kickoff() with run_crew_with_watchdog():")
    print("   result = run_crew_with_watchdog(crew, max_budget_usd=1.0)")
    print()
    print("If the crew loops or overruns budget, WatchdogHalt is raised.")
    print("The issue CrewAI #4495 (infinite tool loop) would be caught by pattern 1.")
