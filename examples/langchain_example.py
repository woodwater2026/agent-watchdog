"""
Agent Watchdog + LangChain integration example.

Shows how to wrap a LangChain agent with loop detection and budget guard.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt

# --- Setup watchdog ---
watchdog = AgentWatchdog(
    max_budget_usd=0.50,       # halt if run exceeds $0.50
    max_identical_calls=3,     # halt if same tool called 3x identically
    timeout_seconds=120,       # halt after 2 minutes
    model="openai/gpt-4o",
)

# --- LangChain setup (standard) ---
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain_openai import ChatOpenAI
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


# --- Option A: Simple wrapping (timeout + budget only) ---
def run_with_watchdog_simple(task: str):
    try:
        with watchdog.watch(run_id=task[:20]):
            # If your agent has a token callback, feed it to watchdog:
            # watchdog.record_tokens(token_in=cb.prompt_tokens, token_out=cb.completion_tokens)
            result = executor.invoke({"input": task})
            return result
    except WatchdogHalt as e:
        print(f"Agent halted: {e.report.reason} after ${e.report.estimated_cost_usd:.4f}")
        return None


# --- Option B: With tool call tracking (loop detection) ---
# Patch your tool to record calls:

def make_watched_tool(original_tool, tool_name: str):
    """Wrap a LangChain tool to record calls to the watchdog."""
    def watched(*args, **kwargs):
        output = original_tool(*args, **kwargs)
        watchdog.record_tool_call(tool_name, args=str(args) + str(kwargs), output=output)
        return output
    watched.__name__ = original_tool.__name__
    return watched


# --- Option C: Using LangChain callbacks for token tracking ---
# from langchain.callbacks import get_openai_callback
#
# def run_with_token_tracking(task: str):
#     with get_openai_callback() as cb:
#         with watchdog.watch(run_id=task[:20]):
#             # Poll token usage periodically (or after each step)
#             result = executor.invoke({"input": task})
#             watchdog.record_tokens(
#                 token_in=cb.prompt_tokens,
#                 token_out=cb.completion_tokens
#             )
#     return result


# --- Demo (without actual LangChain to keep example dependency-free) ---
if __name__ == "__main__":
    print("Agent Watchdog + LangChain demo\n")

    # Simulate a looping agent
    print("Test 1: Loop detection")
    try:
        with watchdog.watch(run_id="demo-loop"):
            for i in range(5):
                watchdog.record_tool_call("search", args="same query every time")
                print(f"  Step {i+1}: called search('same query every time')")
    except WatchdogHalt as e:
        print(f"  ✓ Halted: {e.report.reason} after {len(e.report.tool_calls)} calls\n")

    # Simulate a budget overrun
    print("Test 2: Budget guard")
    watchdog2 = AgentWatchdog(max_budget_usd=0.10, model="openai/gpt-4o")
    try:
        with watchdog2.watch(run_id="demo-budget"):
            watchdog2.record_tokens(token_in=5000, token_out=1000)  # ~$0.135 > $0.10
    except WatchdogHalt as e:
        print(f"  ✓ Halted: {e.report.reason}, cost: ${e.report.estimated_cost_usd:.4f}\n")

    print("Done.")
