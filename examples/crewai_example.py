"""
Agent Watchdog + CrewAI integration example.

CrewAI's main failure mode: tool wrapper regressions that cause infinite loops.
This example shows how to catch that with watchdog.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt, HaltReason

watchdog = AgentWatchdog(
    max_budget_usd=1.0,
    max_identical_calls=3,   # catches the crewai regression pattern
    timeout_seconds=180,
    model="anthropic/claude-sonnet-4-6",
)


# CrewAI tool wrapper pattern — record each tool invocation
class WatchedTool:
    """Mixin for CrewAI tools to report into the watchdog."""

    def _run(self, *args, **kwargs):
        result = self._execute(*args, **kwargs)
        watchdog.record_tool_call(
            tool_name=self.__class__.__name__,
            args=str(args) + str(kwargs),
            output=str(result)[:200],
        )
        return result

    def _execute(self, *args, **kwargs):
        raise NotImplementedError


# Example tool
class SearchTool(WatchedTool):
    name = "search"
    description = "Search for information"

    def _execute(self, query: str) -> str:
        # your implementation
        return f"results for {query}"


# Run a crew with watchdog protection
def run_crew(task: str):
    try:
        with watchdog.watch(run_id="crewai-run"):
            # your crew.kickoff() call goes here
            # the WatchedTool mixin will record each tool call
            tool = SearchTool()
            for _ in range(20):
                tool._run(query=task)         # will loop-detect at call 3+1
                watchdog.record_tokens(800, 200)

    except WatchdogHalt as e:
        r = e.report
        print(f"Crew halted: {r.reason.value} | cost=${r.estimated_cost_usd:.4f} | calls={len(r.tool_calls)}")
        # log to your monitoring system here
        return {"status": "halted", "report": r}


if __name__ == "__main__":
    run_crew("research AI agent infrastructure")
