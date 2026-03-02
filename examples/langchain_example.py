"""
Agent Watchdog + LangChain integration example.

Shows how to wrap a LangChain agent with watchdog protection.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt

# ── LangChain setup (standard) ──────────────────────────────────────────────
# from langchain_openai import ChatOpenAI
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain.tools import tool
#
# @tool
# def search(query: str) -> str:
#     """Search the web."""
#     return f"Results for: {query}"
#
# llm = ChatOpenAI(model="gpt-4o")
# agent = create_react_agent(llm, [search], prompt)
# executor = AgentExecutor(agent=agent, tools=[search], verbose=True)

# ── Watchdog integration ─────────────────────────────────────────────────────

watchdog = AgentWatchdog(
    max_budget_usd=1.0,       # halt if run exceeds $1
    max_identical_calls=3,    # halt if same tool+args called 3x consecutively
    timeout_seconds=120,      # halt after 2 minutes
    model="openai/gpt-4o",
)

def run_with_watchdog(task: str, run_id: str = "langchain-run"):
    try:
        with watchdog.watch(run_id=run_id):
            # Option A: wrap the entire run (timeout + budget only)
            # result = executor.invoke({"input": task})

            # Option B: instrument each step for loop detection
            # Use LangChain callbacks to record tool calls
            result = _run_instrumented(task)
            return result

    except WatchdogHalt as e:
        report = e.report
        print(f"\n⚠️  Agent halted: {report.reason.value}")
        print(f"   Cost: ${report.estimated_cost_usd:.4f}")
        print(f"   Steps: {len(report.tool_calls)}")
        print(f"   Last output: {report.last_output}")
        return {"error": report.reason.value, "report": report}


def _run_instrumented(task: str):
    """
    Example of instrumenting tool calls for loop detection.
    In real LangChain usage, do this via a custom callback handler.
    """
    # Simulate agent steps
    steps = [
        ("search", "AI agent frameworks 2026"),
        ("search", "AI agent frameworks 2026"),   # same args
        ("search", "AI agent frameworks 2026"),   # triggers loop detection
    ]

    for tool_name, args in steps:
        # Call the actual tool here
        output = f"[simulated output for {tool_name}({args})]"

        # Record for watchdog
        watchdog.record_tool_call(tool_name, args=args, output=output)

        # Estimate tokens (or get from LangChain callbacks)
        watchdog.record_tokens(token_in=500, token_out=200)

    return {"output": "completed"}


# ── LangChain Callback Handler (for production use) ──────────────────────────
# class WatchdogCallbackHandler(BaseCallbackHandler):
#     def __init__(self, watchdog: AgentWatchdog):
#         self.watchdog = watchdog
#
#     def on_tool_start(self, serialized, input_str, **kwargs):
#         tool_name = serialized.get("name", "unknown")
#         self.watchdog.record_tool_call(tool_name, args=input_str)
#
#     def on_llm_end(self, response, **kwargs):
#         usage = response.llm_output.get("token_usage", {})
#         self.watchdog.record_tokens(
#             token_in=usage.get("prompt_tokens", 0),
#             token_out=usage.get("completion_tokens", 0),
#         )


if __name__ == "__main__":
    print("Running instrumented agent with watchdog...")
    result = run_with_watchdog("Research AI agent frameworks", run_id="demo-001")
    print(f"Result: {result}")
