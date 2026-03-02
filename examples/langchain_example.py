"""
Agent Watchdog + LangChain integration example.

Shows how to wrap a LangChain agent with loop detection and budget guard.
The key is using a custom callback to feed tool call data into the watchdog.
"""
from langchain.agents import AgentExecutor
from langchain.callbacks.base import BaseCallbackHandler
from agent_watchdog import AgentWatchdog, WatchdogHalt


class WatchdogCallback(BaseCallbackHandler):
    """LangChain callback that feeds tool usage into AgentWatchdog."""

    def __init__(self, watchdog: AgentWatchdog):
        self.watchdog = watchdog

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown_tool")
        self.watchdog.record_tool_call(tool_name, args=input_str)

    def on_tool_end(self, output, **kwargs):
        # Update last output
        if self.watchdog._current_run:
            self.watchdog._current_run.last_output = str(output)[:500]

    def on_llm_end(self, response, **kwargs):
        # Record token usage from LLM response
        usage = getattr(response, "llm_output", {}) or {}
        token_usage = usage.get("token_usage", {})
        self.watchdog.record_tokens(
            token_in=token_usage.get("prompt_tokens", 0),
            token_out=token_usage.get("completion_tokens", 0),
        )


def run_with_watchdog(agent_executor: AgentExecutor, task: str, run_id: str = "run"):
    """
    Run a LangChain AgentExecutor with watchdog protection.

    Args:
        agent_executor: Your configured LangChain AgentExecutor
        task: The task string to run
        run_id: Identifier for this run (for logging)

    Returns:
        dict with 'output' on success, 'halted' and 'report' on WatchdogHalt

    Example:
        result = run_with_watchdog(agent, "research the top 5 AI papers this week")
        if result.get("halted"):
            print(f"Agent halted: {result['report'].reason}")
        else:
            print(result["output"])
    """
    watchdog = AgentWatchdog(
        max_budget_usd=1.0,
        max_identical_calls=3,
        timeout_seconds=120,
    )
    callback = WatchdogCallback(watchdog)

    try:
        with watchdog.watch(run_id=run_id):
            result = agent_executor.invoke(
                {"input": task},
                config={"callbacks": [callback]},
            )
        return {"output": result.get("output"), "halted": False}

    except WatchdogHalt as e:
        return {
            "halted": True,
            "report": e.report,
            "partial_output": e.report.last_output,
        }


# --- Minimal runnable demo (requires langchain + openai) ---
if __name__ == "__main__":
    try:
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain_core.tools import tool
        from langchain import hub

        call_count = 0

        @tool
        def broken_search(query: str) -> str:
            """Search for information. Always returns an error."""
            global call_count
            call_count += 1
            return "Error: search service unavailable"

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, [broken_search], prompt)
        executor = AgentExecutor(agent=agent, tools=[broken_search], verbose=False)

        print("Running agent with a broken tool (should loop and be halted)...")
        result = run_with_watchdog(executor, "find information about LangChain", run_id="demo")

        if result["halted"]:
            r = result["report"]
            print(f"✓ Watchdog halted the agent: {r.reason.value}")
            print(f"  Tool calls made: {len(r.tool_calls)}")
            print(f"  Estimated cost: ${r.estimated_cost_usd:.4f}")
            print(f"  Elapsed: {r.elapsed_seconds:.1f}s")
        else:
            print(f"Agent completed: {result['output']}")

    except ImportError as e:
        print(f"Demo requires langchain-openai: {e}")
        print("Install with: pip install langchain langchain-openai")
