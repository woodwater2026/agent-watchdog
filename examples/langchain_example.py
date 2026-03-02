"""
Agent Watchdog + LangChain example.

Shows how to wrap a LangChain agent with loop detection, budget guards, and timeout.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt, HaltReason

# --- Setup LangChain agent (replace with your actual agent) ---
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain_openai import ChatOpenAI
# ...

# --- Minimal mock for demo ---
class MockLangChainAgent:
    """Simulates an agent that gets stuck in a loop."""
    def __init__(self, loop=False):
        self.loop = loop
        self._calls = 0

    def run(self, task: str, watchdog: AgentWatchdog = None):
        while True:
            tool = "search"
            args = "same query" if self.loop else f"query-{self._calls}"
            self._calls += 1

            # Simulate tool execution
            output = f"Result for {args}"

            # Record with watchdog (triggers loop detection)
            if watchdog:
                watchdog.record_tool_call(tool, args=args, output=output)
                watchdog.record_tokens(token_in=500, token_out=200)

            if not self.loop and self._calls >= 3:
                return "Task complete"


# --- Normal run ---
print("=== Normal run ===")
watchdog = AgentWatchdog(max_budget_usd=1.0, max_identical_calls=3, timeout_seconds=30)
agent = MockLangChainAgent(loop=False)

try:
    with watchdog.watch(run_id="normal-run"):
        result = agent.run("do the task", watchdog=watchdog)
    print(f"Result: {result}")
except WatchdogHalt as e:
    print(f"Halted: {e.report}")


# --- Loop run ---
print("\n=== Loop detection ===")
watchdog2 = AgentWatchdog(max_budget_usd=1.0, max_identical_calls=3, timeout_seconds=30)
agent2 = MockLangChainAgent(loop=True)

try:
    with watchdog2.watch(run_id="loop-run"):
        result = agent2.run("do the task", watchdog=watchdog2)
    print(f"Result: {result}")
except WatchdogHalt as e:
    r = e.report
    print(f"Caught loop! reason={r.reason} calls={len(r.tool_calls)} cost=${r.estimated_cost_usd:.4f}")
    assert r.reason == HaltReason.LOOP_DETECTED

print("\nDone.")
