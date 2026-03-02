"""
Agent Watchdog + LangChain integration example.

Shows how to wrap a LangChain agent with watchdog protection.
Requires: pip install agent-watchdog langchain langchain-openai
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt, HaltReason


# --- Option A: Manual instrumentation via callbacks ---

from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List, Union


class WatchdogCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback that feeds tool calls and token usage into AgentWatchdog.
    Attach this to any LangChain agent or LLM.
    """

    def __init__(self, watchdog: AgentWatchdog):
        self.watchdog = watchdog

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        tool_name = serialized.get("name", "unknown_tool")
        self.watchdog.record_tool_call(tool_name, args=input_str)

    def on_tool_end(self, output: str, **kwargs):
        # Update last output in current run state
        if self.watchdog._current_run:
            self.watchdog._current_run.last_output = output[:500]

    def on_llm_end(self, response, **kwargs):
        # Record token usage if available
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self.watchdog.record_tokens(
                token_in=usage.get("prompt_tokens", 0),
                token_out=usage.get("completion_tokens", 0),
            )


# --- Usage ---

def run_with_watchdog(agent, task: str, run_id: str = "langchain-run"):
    """
    Run a LangChain agent with watchdog protection.
    Returns result or raises WatchdogHalt.
    """
    watchdog = AgentWatchdog(
        max_budget_usd=1.0,
        max_identical_calls=3,
        timeout_seconds=120,
    )
    callback = WatchdogCallbackHandler(watchdog)

    try:
        with watchdog.watch(run_id=run_id):
            result = agent.invoke(
                {"input": task},
                config={"callbacks": [callback]},
            )
        return result
    except WatchdogHalt as e:
        report = e.report
        print(f"\n[Watchdog] Agent halted: {report.reason.value}")
        print(f"  Cost: ${report.estimated_cost_usd:.4f}")
        print(f"  Calls: {len(report.tool_calls)}")
        print(f"  Time: {report.elapsed_seconds:.1f}s")
        print(f"  Reason: {report.message}")
        raise


# --- Minimal runnable demo (no real API key needed for structure test) ---

if __name__ == "__main__":
    print("Agent Watchdog + LangChain — callback integration")
    print("To run with a real agent:")
    print("  from langchain_openai import ChatOpenAI")
    print("  from langchain.agents import create_react_agent, AgentExecutor")
    print("  llm = ChatOpenAI(model='gpt-4o-mini')")
    print("  # ... create agent ...")
    print("  result = run_with_watchdog(agent_executor, 'your task')")
    print()
    print("The WatchdogCallbackHandler records every tool call.")
    print("If the agent loops or exceeds budget, WatchdogHalt is raised with a full report.")
