"""
Agent Watchdog + LangChain integration example.

This shows how to use AgentWatchdog with a LangChain agent by wrapping
the agent executor and recording tool calls via a custom callback.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt

# --- Setup ---
watchdog = AgentWatchdog(
    max_budget_usd=0.50,
    max_identical_calls=3,
    timeout_seconds=120,
    model="openai/gpt-4o",
)

# --- LangChain callback to feed watchdog ---
try:
    from langchain.callbacks.base import BaseCallbackHandler

    class WatchdogCallback(BaseCallbackHandler):
        def __init__(self, watchdog: AgentWatchdog):
            self.watchdog = watchdog

        def on_tool_start(self, serialized, input_str, **kwargs):
            tool_name = serialized.get("name", "unknown_tool")
            self.watchdog.record_tool_call(tool_name, args=input_str)

        def on_llm_end(self, response, **kwargs):
            # Approximate token usage from LLM response
            usage = getattr(response, "llm_output", {}) or {}
            token_usage = usage.get("token_usage", {})
            self.watchdog.record_tokens(
                token_in=token_usage.get("prompt_tokens", 0),
                token_out=token_usage.get("completion_tokens", 0),
            )

except ImportError:
    print("LangChain not installed. Install with: pip install langchain")
    WatchdogCallback = None


# --- Usage ---
def run_with_watchdog(agent_executor, task: str, run_id: str = "run"):
    """
    Run a LangChain agent executor with watchdog protection.
    Returns (result, None) on success, (None, report) on halt.
    """
    if WatchdogCallback is None:
        raise ImportError("LangChain required")

    callback = WatchdogCallback(watchdog)

    try:
        with watchdog.watch(run_id=run_id):
            result = agent_executor.invoke(
                {"input": task},
                config={"callbacks": [callback]},
            )
            return result, None
    except WatchdogHalt as e:
        print(f"Agent halted: {e.report.reason} after ${e.report.estimated_cost_usd:.4f}")
        return None, e.report


# --- Example (requires langchain + openai) ---
if __name__ == "__main__":
    print("LangChain + AgentWatchdog example.")
    print("Configure your agent executor and call run_with_watchdog(executor, task).")
    print("See README for full setup.")
