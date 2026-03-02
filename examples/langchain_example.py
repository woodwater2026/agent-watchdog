"""
Agent Watchdog + LangChain integration example.

Pattern: wrap the agent executor run and record tool calls via callbacks.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt

# --- Setup watchdog ---
watchdog = AgentWatchdog(
    max_budget_usd=0.50,
    max_identical_calls=3,
    timeout_seconds=120,
    model="openai/gpt-4o",
)

# --- LangChain callback to feed watchdog ---
try:
    from langchain_core.callbacks import BaseCallbackHandler

    class WatchdogCallback(BaseCallbackHandler):
        """Feeds LangChain tool calls into AgentWatchdog."""

        def __init__(self, wd: AgentWatchdog):
            self.wd = wd

        def on_tool_start(self, serialized, input_str, **kwargs):
            tool_name = serialized.get("name", "unknown_tool")
            self.wd.record_tool_call(tool_name, args=input_str)

        def on_tool_end(self, output, **kwargs):
            pass  # output recorded via on_tool_start for now

        def on_llm_end(self, response, **kwargs):
            # Record token usage if available
            usage = getattr(response, "llm_output", {}) or {}
            token_usage = usage.get("token_usage", {})
            if token_usage:
                self.wd.record_tokens(
                    token_in=token_usage.get("prompt_tokens", 0),
                    token_out=token_usage.get("completion_tokens", 0),
                )

except ImportError:
    print("langchain_core not installed. Install with: pip install langchain-core")
    WatchdogCallback = None


# --- Example usage ---
def run_agent_with_watchdog(agent_executor, task: str, run_id: str = "run"):
    """
    Run a LangChain AgentExecutor with watchdog protection.

    Args:
        agent_executor: A LangChain AgentExecutor instance
        task: The task string to pass to the agent
        run_id: Identifier for this run (for logs)

    Returns:
        The agent result, or None if halted

    Example:
        result = run_agent_with_watchdog(agent, "search for recent AI papers", "research-001")
    """
    callback = WatchdogCallback(watchdog) if WatchdogCallback else None
    callbacks = [callback] if callback else []

    try:
        with watchdog.watch(run_id=run_id):
            result = agent_executor.invoke(
                {"input": task},
                config={"callbacks": callbacks},
            )
            return result
    except WatchdogHalt as e:
        r = e.report
        print(f"\n[HALTED] reason={r.reason.value} cost=${r.estimated_cost_usd:.4f} "
              f"after {r.elapsed_seconds:.1f}s and {len(r.tool_calls)} tool calls")
        print(f"Last output: {r.last_output}")
        return None


if __name__ == "__main__":
    print("Agent Watchdog LangChain example")
    print("See README for full setup instructions.")
    print("pip install agent-watchdog langchain langchain-openai")
