"""
Agent Watchdog + CrewAI integration example.

CrewAI doesn't expose a direct callback interface for individual tool calls,
so we wrap at the crew.kickoff() level for budget and timeout protection.
For loop detection, subclass BaseTool.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt

watchdog = AgentWatchdog(
    max_budget_usd=1.0,
    max_identical_calls=3,
    timeout_seconds=300,
    model="anthropic/claude-sonnet-4-6",
)

# --- Tool wrapper for loop detection ---
try:
    from crewai.tools import BaseTool

    class WatchdogTool(BaseTool):
        """
        Mixin: wrap any CrewAI tool to report calls to watchdog.

        Usage:
            class MySearchTool(WatchdogTool):
                name = "search"
                description = "Search the web"
                _watchdog: AgentWatchdog = None

                def _run(self, query: str) -> str:
                    return search_api(query)

            tool = MySearchTool()
            tool._watchdog = watchdog
        """
        _watchdog: object = None

        def _run(self, *args, **kwargs):
            raise NotImplementedError

        def run(self, *args, **kwargs):
            if self._watchdog:
                self._watchdog.record_tool_call(
                    self.name,
                    args=str(args) + str(kwargs),
                )
            return super().run(*args, **kwargs)

except ImportError:
    print("crewai not installed. Install with: pip install crewai")
    WatchdogTool = None


# --- Example usage ---
def run_crew_with_watchdog(crew, run_id: str = "crew-run"):
    """
    Run a CrewAI Crew with watchdog timeout and budget protection.

    Args:
        crew: A CrewAI Crew instance
        run_id: Identifier for this run

    Returns:
        Crew result, or None if halted

    Example:
        result = run_crew_with_watchdog(my_crew, "research-crew-001")
    """
    try:
        with watchdog.watch(run_id=run_id):
            result = crew.kickoff()
            return result
    except WatchdogHalt as e:
        r = e.report
        print(f"\n[CREW HALTED] reason={r.reason.value} "
              f"cost=${r.estimated_cost_usd:.4f} after {r.elapsed_seconds:.1f}s")
        return None


if __name__ == "__main__":
    print("Agent Watchdog CrewAI example")
    print("pip install agent-watchdog crewai")
