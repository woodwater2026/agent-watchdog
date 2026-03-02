"""
Agent Watchdog + LangChain integration example.

Shows how to wrap a LangChain agent with watchdog monitoring.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt, HaltReason

# --- Setup watchdog ---
watchdog = AgentWatchdog(
    max_budget_usd=0.50,       # halt if run exceeds $0.50
    max_identical_calls=3,     # halt if same tool called 3x identically
    timeout_seconds=120,       # halt after 2 minutes
    model="openai/gpt-4o",
)

# --- LangChain tool callback ---
# Wrap your tools to record calls into the watchdog
from functools import wraps

def watched_tool(tool_fn):
    """Decorator to record LangChain tool calls into the watchdog."""
    @wraps(tool_fn)
    def wrapper(*args, **kwargs):
        result = tool_fn(*args, **kwargs)
        # Record the call — watchdog checks for loops here
        watchdog.record_tool_call(
            tool_name=tool_fn.__name__,
            args=str(args) + str(kwargs),
            output=str(result)[:200],
        )
        return result
    return wrapper


# --- Example: wrap a search tool ---
@watched_tool
def search(query: str) -> str:
    # your actual search implementation
    return f"Results for: {query}"


# --- Run the agent ---
def run_agent(task: str):
    try:
        with watchdog.watch(run_id=f"lc-{task[:20]}"):
            # Simulate a LangChain agent loop
            for i in range(10):
                result = search(task)   # watchdog records this
                watchdog.record_tokens(token_in=500, token_out=100)
                # agent decides what to do next...
                if "done" in result:
                    break
            return result

    except WatchdogHalt as e:
        report = e.report
        print(f"\n[Agent stopped] reason={report.reason.value}")
        print(f"  cost=${report.estimated_cost_usd:.4f}")
        print(f"  calls={len(report.tool_calls)}")
        print(f"  elapsed={report.elapsed_seconds:.1f}s")
        if report.reason == HaltReason.LOOP_DETECTED:
            print("  → Agent was stuck in a loop. Check your tool logic.")
        elif report.reason == HaltReason.BUDGET_EXCEEDED:
            print("  → Run exceeded budget. Consider breaking the task into smaller pieces.")
        return None


if __name__ == "__main__":
    run_agent("find information about AI agent reliability")
