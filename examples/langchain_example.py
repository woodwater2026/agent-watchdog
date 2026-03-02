"""
Agent Watchdog + LangChain integration example.

Shows how to wrap a LangChain agent with AgentWatchdog for:
- Loop detection (same tool call repeated)
- Budget guard (per-run cost cap)
- Graceful halt with structured report

Requirements:
    pip install agent-watchdog langchain langchain-openai
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt

# --- Minimal integration via callback ---
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Union


class WatchdogCallback(BaseCallbackHandler):
    """LangChain callback that feeds tool calls and token usage to AgentWatchdog."""

    def __init__(self, watchdog: AgentWatchdog):
        self.watchdog = watchdog

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        tool_name = serialized.get("name", "unknown_tool")
        self.watchdog.record_tool_call(tool_name, args=input_str)

    def on_tool_end(self, output: str, **kwargs):
        # Update last output on the current run state
        if self.watchdog._current_run:
            self.watchdog._current_run.last_output = str(output)[:500]

    def on_llm_end(self, response: LLMResult, **kwargs):
        # Record token usage if available
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.watchdog.record_tokens(
                token_in=usage.get("prompt_tokens", 0),
                token_out=usage.get("completion_tokens", 0),
            )


# --- Usage example ---
def run_agent_with_watchdog(agent, task: str, run_id: str = "run"):
    """
    Run a LangChain agent under watchdog protection.

    Returns (result, report) where report is None on success,
    or a HaltReport if the watchdog intervened.
    """
    watchdog = AgentWatchdog(
        max_budget_usd=1.0,
        max_identical_calls=3,
        timeout_seconds=120,
        model="openai/gpt-4o",
    )
    callback = WatchdogCallback(watchdog)

    try:
        with watchdog.watch(run_id=run_id):
            result = agent.invoke(
                {"input": task},
                config={"callbacks": [callback]},
            )
        return result, None

    except WatchdogHalt as e:
        report = e.report
        print(f"\n[Watchdog] Run halted: {report.reason.value}")
        print(f"  Cost so far: ${report.estimated_cost_usd:.4f}")
        print(f"  Tool calls: {len(report.tool_calls)}")
        print(f"  Last output: {report.last_output}")
        return None, report


# --- Demo (requires OpenAI key) ---
if __name__ == "__main__":
    import os
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_core.tools import tool
    from langchain import hub

    call_count = 0

    @tool
    def broken_search(query: str) -> str:
        """Search the web for information."""
        global call_count
        call_count += 1
        # Simulate a broken tool that always returns the same unhelpful result
        return "No results found. Try again."

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, [broken_search], prompt)
    executor = AgentExecutor(agent=agent, tools=[broken_search], verbose=True, max_iterations=20)

    print("Running agent with a broken tool (will loop)...")
    result, report = run_agent_with_watchdog(executor, "Find the latest news about AI agents", run_id="demo-loop")

    if report:
        print(f"\nWatchdog caught it: {report.reason.value} after {len(report.tool_calls)} calls")
    else:
        print(f"\nAgent completed: {result}")
