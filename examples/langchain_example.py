"""
Agent Watchdog + LangChain integration example.

Shows how to wrap a LangChain agent with watchdog protection.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt, HaltReason

# --- LangChain setup (standard) ---
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain_openai import ChatOpenAI
# from langchain.tools import tool
#
# @tool
# def search(query: str) -> str:
#     """Search the web for information."""
#     return f"Results for: {query}"
#
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
# agent_executor = AgentExecutor(agent=..., tools=[search], verbose=True)

# --- Watchdog integration ---

watchdog = AgentWatchdog(
    max_budget_usd=0.50,        # halt if this run exceeds $0.50
    max_identical_calls=3,      # halt if same tool called 3x with same args
    timeout_seconds=120,        # halt after 2 minutes
    model="openai/gpt-4o",
)

# Option 1: Wrap the entire run (simplest, no per-call visibility)
def run_with_watchdog_simple(task: str):
    try:
        with watchdog.watch(run_id=f"langchain-{hash(task)}"):
            # result = agent_executor.invoke({"input": task})
            result = {"output": "demo result"}  # placeholder
            return result
    except WatchdogHalt as e:
        print(f"Agent halted: {e.report.reason}")
        print(f"Cost incurred: ${e.report.estimated_cost_usd:.4f}")
        return {"output": None, "halt_reason": e.report.reason}


# Option 2: Use LangChain callbacks for per-tool visibility (recommended)
# from langchain.callbacks.base import BaseCallbackHandler
#
# class WatchdogCallback(BaseCallbackHandler):
#     def __init__(self, watchdog: AgentWatchdog):
#         self.watchdog = watchdog
#
#     def on_tool_start(self, serialized, input_str, **kwargs):
#         tool_name = serialized.get("name", "unknown")
#         self.watchdog.record_tool_call(tool_name, args=input_str)
#
#     def on_llm_end(self, response, **kwargs):
#         # Record token usage from LLM response
#         usage = response.llm_output.get("token_usage", {})
#         self.watchdog.record_tokens(
#             token_in=usage.get("prompt_tokens", 0),
#             token_out=usage.get("completion_tokens", 0),
#         )
#
# def run_with_watchdog_full(task: str):
#     try:
#         with watchdog.watch(run_id=f"langchain-{hash(task)}"):
#             cb = WatchdogCallback(watchdog)
#             result = agent_executor.invoke(
#                 {"input": task},
#                 config={"callbacks": [cb]}
#             )
#             return result
#     except WatchdogHalt as e:
#         return {"output": None, "halt_report": e.report}


if __name__ == "__main__":
    result = run_with_watchdog_simple("What is the weather in New York?")
    print(result)
