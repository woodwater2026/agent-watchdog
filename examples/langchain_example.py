"""
Agent Watchdog + LangChain integration example.

Shows how to wrap a LangChain agent with Agent Watchdog for:
- Loop detection (via custom callback)
- Budget guard
- Timeout enforcement
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt, HaltReason

# LangChain imports (optional dependency)
try:
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import AgentAction, AgentFinish, LLMResult
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Install langchain to use this integration: pip install langchain")


if LANGCHAIN_AVAILABLE:
    class WatchdogCallback(BaseCallbackHandler):
        """
        LangChain callback handler that feeds data into AgentWatchdog.
        Pass this to your LangChain agent to enable loop detection and cost tracking.
        """

        def __init__(self, watchdog: AgentWatchdog):
            self.watchdog = watchdog

        def on_agent_action(self, action: AgentAction, **kwargs):
            """Called when the agent decides to use a tool."""
            self.watchdog.record_tool_call(
                tool_name=action.tool,
                args=action.tool_input,
            )

        def on_llm_end(self, response: LLMResult, **kwargs):
            """Called after each LLM call — record token usage."""
            for gen in response.generations:
                for g in gen:
                    if hasattr(g, 'generation_info') and g.generation_info:
                        usage = g.generation_info.get('usage', {})
                        self.watchdog.record_tokens(
                            token_in=usage.get('prompt_tokens', 0),
                            token_out=usage.get('completion_tokens', 0),
                        )

        def on_tool_end(self, output: str, **kwargs):
            """Called after a tool returns — update last output."""
            if self.watchdog._current_run:
                self.watchdog._current_run.last_output = output[:500]


# --- Usage example ---

def run_agent_with_watchdog(agent, task: str, run_id: str = "run"):
    """
    Run a LangChain agent with watchdog protection.

    Args:
        agent: A LangChain AgentExecutor
        task: The task string to run
        run_id: Identifier for this run (for logging)

    Returns:
        Agent output, or None if halted
    """
    watchdog = AgentWatchdog(
        max_budget_usd=0.50,      # halt if estimated cost > $0.50
        max_identical_calls=3,    # halt if same tool called 3x identically
        timeout_seconds=120,      # halt after 2 minutes
        model="openai/gpt-4o",
    )

    callback = WatchdogCallback(watchdog)

    try:
        with watchdog.watch(run_id=run_id):
            result = agent.run(task, callbacks=[callback])
        return result

    except WatchdogHalt as e:
        r = e.report
        print(f"\n[Watchdog halted run '{r.run_id}']")
        print(f"  Reason:  {r.reason.value}")
        print(f"  Cost:    ${r.estimated_cost_usd:.4f}")
        print(f"  Elapsed: {r.elapsed_seconds:.1f}s")
        print(f"  Calls:   {len(r.tool_calls)}")
        if r.last_output:
            print(f"  Last:    {r.last_output[:100]}")
        return None


# Minimal runnable demo (no real LangChain agent needed)
if __name__ == "__main__":
    from agent_watchdog import AgentWatchdog, WatchdogHalt

    print("Demo: simulating a looping agent\n")

    watchdog = AgentWatchdog(max_identical_calls=3, timeout_seconds=10)

    try:
        with watchdog.watch(run_id="demo-loop"):
            for i in range(10):
                print(f"  Step {i+1}: calling search tool...")
                watchdog.record_tool_call("search", args="the same query every time")
    except WatchdogHalt as e:
        print(f"\nHalted! Reason: {e.report.reason.value}")
        print(f"Total calls before halt: {len(e.report.tool_calls)}")

    print("\nDemo: simulating budget overrun\n")

    watchdog2 = AgentWatchdog(max_budget_usd=0.01, model="anthropic/claude-sonnet-4-6")

    try:
        with watchdog2.watch(run_id="demo-budget"):
            # Simulate a run that uses many tokens
            watchdog2.record_tokens(token_in=5000, token_out=500)
    except WatchdogHalt as e:
        print(f"Halted! Reason: {e.report.reason.value}")
        print(f"Estimated cost: ${e.report.estimated_cost_usd:.4f}")
