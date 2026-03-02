"""
Agent Watchdog + LangChain integration example.

Shows how to wrap a LangChain agent with AgentWatchdog for loop detection,
budget guards, and graceful halts.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt

# ── LangChain imports (install: pip install langchain langchain-openai) ──────
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain_openai import ChatOpenAI
# from langchain.tools import tool

# ── Example: wrapping an AgentExecutor ───────────────────────────────────────

def run_with_watchdog(agent_executor, task: str, run_id: str = "lc-run"):
    """
    Run a LangChain AgentExecutor under AgentWatchdog supervision.
    
    The watchdog hooks into LangChain's callback system to:
    - Record each tool call (for loop detection)
    - Track token usage (for budget enforcement)
    - Enforce a timeout
    """
    watchdog = AgentWatchdog(
        max_budget_usd=1.0,
        max_identical_calls=3,
        timeout_seconds=120,
        model="openai/gpt-4o",
    )

    class WatchdogCallback:
        """LangChain callback handler that feeds data to the watchdog."""

        def on_tool_start(self, serialized, input_str, **kwargs):
            tool_name = serialized.get("name", "unknown")
            watchdog.record_tool_call(tool_name, args=input_str)

        def on_tool_end(self, output, **kwargs):
            pass  # output recorded on next tool_start via last_output

        def on_llm_end(self, response, **kwargs):
            # Record token usage from LangChain's LLMResult
            if hasattr(response, "llm_output") and response.llm_output:
                usage = response.llm_output.get("token_usage", {})
                watchdog.record_tokens(
                    token_in=usage.get("prompt_tokens", 0),
                    token_out=usage.get("completion_tokens", 0),
                )

    try:
        with watchdog.watch(run_id=run_id):
            result = agent_executor.invoke(
                {"input": task},
                config={"callbacks": [WatchdogCallback()]},
            )
            return result
    except WatchdogHalt as e:
        report = e.report
        print(f"\n⚠️  Agent halted: {report.reason.value}")
        print(f"   Cost incurred: ${report.estimated_cost_usd:.4f}")
        print(f"   Tool calls made: {len(report.tool_calls)}")
        print(f"   Last output: {report.last_output}")
        return {"error": report.reason.value, "report": report}


# ── Minimal demo (no real LLM needed) ────────────────────────────────────────

def demo_loop_detection():
    """Simulates an agent stuck in a loop."""
    watchdog = AgentWatchdog(max_identical_calls=3, timeout_seconds=10)

    try:
        with watchdog.watch(run_id="demo-loop"):
            # Simulate an agent calling the same tool 3x with same args
            for i in range(5):
                watchdog.record_tool_call("web_search", args="latest AI news")
                print(f"  Step {i+1}: called web_search('latest AI news')")
    except WatchdogHalt as e:
        print(f"\n✓ Loop detected and halted: {e.report.message}")
        print(f"  Calls before halt: {len(e.report.tool_calls)}")


def demo_budget_guard():
    """Simulates a run exceeding budget."""
    watchdog = AgentWatchdog(
        max_budget_usd=0.05,
        model="openai/gpt-4o",
    )

    try:
        with watchdog.watch(run_id="demo-budget"):
            watchdog.record_tokens(token_in=5000, token_out=1000)   # ~$0.022
            print("  Step 1: 5k in / 1k out tokens logged")
            watchdog.record_tokens(token_in=5000, token_out=1000)   # ~$0.044 total → over $0.05
            print("  Step 2: 5k in / 1k out tokens logged")
    except WatchdogHalt as e:
        print(f"\n✓ Budget exceeded: {e.report.message}")
        print(f"  Estimated cost: ${e.report.estimated_cost_usd:.4f}")


if __name__ == "__main__":
    print("=== Demo: Loop Detection ===")
    demo_loop_detection()

    print("\n=== Demo: Budget Guard ===")
    demo_budget_guard()

    print("\n=== LangChain integration: see run_with_watchdog() above ===")
    print("Usage:")
    print("  result = run_with_watchdog(agent_executor, 'your task here')")
