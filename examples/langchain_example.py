"""
Agent Watchdog + LangChain integration example.

Shows how to wrap a LangChain agent with watchdog protection.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt

# --- Setup (replace with your actual LangChain agent) ---
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain_openai import ChatOpenAI
# agent_executor = AgentExecutor(agent=..., tools=[...])

# --- Watchdog wrapper ---
watchdog = AgentWatchdog(
    max_budget_usd=0.50,      # halt if this run exceeds $0.50
    max_identical_calls=3,    # halt if same tool called 3x identically
    timeout_seconds=120,      # halt after 2 minutes
    model="openai/gpt-4o",
)

def run_with_watchdog(task: str, run_id: str = "langchain-run"):
    try:
        with watchdog.watch(run_id=run_id):
            # Option A: simple wrap (timeout + budget only)
            result = agent_executor.invoke({"input": task})
            return result

    except WatchdogHalt as e:
        report = e.report
        print(f"\n[HALTED] reason={report.reason.value}")
        print(f"  elapsed: {report.elapsed_seconds:.1f}s")
        print(f"  cost: ${report.estimated_cost_usd:.4f}")
        print(f"  calls: {len(report.tool_calls)}")
        print(f"  last output: {report.last_output}")
        return None


# --- Option B: record tool calls for loop detection ---
# Patch the agent's tool-calling mechanism to record each call.
# LangChain uses callbacks for this.

from typing import Any, Union
# from langchain.callbacks.base import BaseCallbackHandler

class WatchdogCallback:  # (BaseCallbackHandler)
    """LangChain callback that feeds tool calls into the watchdog."""

    def __init__(self, wd: AgentWatchdog):
        self.wd = wd

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        self.wd.record_tool_call(tool_name, args=input_str)

    def on_llm_end(self, response: Any, **kwargs):
        # Approximate token recording from LangChain response
        usage = getattr(response, "llm_output", {}).get("token_usage", {})
        self.wd.record_tokens(
            token_in=usage.get("prompt_tokens", 0),
            token_out=usage.get("completion_tokens", 0),
        )


# Usage with callback:
# callback = WatchdogCallback(watchdog)
# with watchdog.watch(run_id="my-run"):
#     result = agent_executor.invoke(
#         {"input": task},
#         config={"callbacks": [callback]}
#     )
