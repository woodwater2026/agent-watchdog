"""
Agent Watchdog + CrewAI integration example.

CrewAI issue #4495: tool wrapper regression causes infinite tool-call loop.
This example shows how to wrap CrewAI tool execution with watchdog protection.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt

watchdog = AgentWatchdog(
    max_budget_usd=1.00,
    max_identical_calls=3,
    timeout_seconds=300,
    model="anthropic/claude-sonnet-4-6",
)


# --- Option A: Wrap the entire crew.kickoff() call ---
def run_crew_with_watchdog(crew, inputs: dict):
    """
    Simplest integration: wrap the full crew run.
    Catches runaway loops via timeout. Does not do per-tool loop detection.
    """
    try:
        with watchdog.watch(run_id="crew-run"):
            result = crew.kickoff(inputs=inputs)
            return result
    except WatchdogHalt as e:
        print(f"[Watchdog] Crew halted: {e.report.reason}")
        print(f"[Watchdog] Elapsed: {e.report.elapsed_seconds:.1f}s")
        print(f"[Watchdog] Estimated cost: ${e.report.estimated_cost_usd:.4f}")
        return None


# --- Option B: Wrap individual BaseTool._run() for loop detection ---
# This is the pattern that CrewAI issue #4495 broke.
# Wrap your tools to report to watchdog:

# from crewai.tools import BaseTool
#
# class WatchedTool(BaseTool):
#     name: str = "watched_tool"
#     description: str = "A tool monitored by Agent Watchdog"
#
#     def _run(self, query: str) -> str:
#         result = self._do_work(query)
#         watchdog.record_tool_call(self.name, args=query, output=result)
#         return result
#
#     def _do_work(self, query: str) -> str:
#         # actual tool logic here
#         return f"result for {query}"


# --- Demo ---
if __name__ == "__main__":
    print("Agent Watchdog + CrewAI demo\n")

    # Simulate CrewAI issue #4495: tool called with no args forever
    print("Simulating issue #4495: infinite tool loop with empty args")
    try:
        with watchdog.watch(run_id="crewai-bug-4495"):
            for i in range(10):
                # Tool called with no meaningful args — this is what the bug causes
                watchdog.record_tool_call("BedrockKBRetrieverTool", args=None)
                print(f"  Tool call {i+1} (no args)")
    except WatchdogHalt as e:
        print(f"\n  ✓ Would have been caught: {e.report.reason}")
        print(f"  ✓ After {len(e.report.tool_calls)} calls")
        print(f"  ✓ {e.report.message}\n")

    print("With Agent Watchdog, this loop is caught at call 3 instead of running forever.")
