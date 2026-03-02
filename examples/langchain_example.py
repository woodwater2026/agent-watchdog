"""
Agent Watchdog + LangChain integration example.

Shows how to wrap a LangChain agent with AgentWatchdog for:
- Loop detection via tool callback
- Real-time budget tracking
- Graceful halt on overrun or loop
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt

# --- LangChain setup (standard) ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic


@tool
def search(query: str) -> str:
    """Search the web for information."""
    # Replace with real search tool
    return f"Results for: {query}"


@tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    try:
        with open(path) as f:
            return f.read()[:1000]
    except Exception as e:
        return f"Error: {e}"


# --- Watchdog-aware callback ---
class WatchdogCallback:
    """
    LangChain callback that reports tool calls and token usage to AgentWatchdog.
    Pass as callbacks=[WatchdogCallback(watchdog)] to AgentExecutor.
    """

    def __init__(self, watchdog: AgentWatchdog):
        self.watchdog = watchdog

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        self.watchdog.record_tool_call(tool_name, args=input_str)

    def on_tool_end(self, output, **kwargs):
        pass  # output recorded in on_tool_start via watchdog

    def on_llm_end(self, response, **kwargs):
        # Extract token usage if available
        usage = getattr(response, "llm_output", {}) or {}
        token_usage = usage.get("token_usage", {})
        self.watchdog.record_tokens(
            token_in=token_usage.get("prompt_tokens", 0),
            token_out=token_usage.get("completion_tokens", 0),
        )


# --- Main usage ---
def run_agent_with_watchdog(task: str, run_id: str = "langchain-run"):
    llm = ChatAnthropic(model="claude-haiku-4-5")
    tools = [search, read_file]

    prompt = PromptTemplate.from_template(
        "Answer the following: {input}\n\nThought: {agent_scratchpad}"
    )
    agent = create_react_agent(llm, tools, prompt)

    watchdog = AgentWatchdog(
        max_budget_usd=0.50,       # halt if run exceeds $0.50
        max_identical_calls=3,     # halt on 3 identical tool calls
        timeout_seconds=120,       # halt after 2 minutes
        model="anthropic/claude-haiku-4-5",
    )
    callback = WatchdogCallback(watchdog)

    executor = AgentExecutor(agent=agent, tools=tools, callbacks=[callback], verbose=False)

    try:
        with watchdog.watch(run_id=run_id):
            result = executor.invoke({"input": task})
        return result["output"]

    except WatchdogHalt as e:
        r = e.report
        print(f"Agent halted: {r.reason} after {r.elapsed_seconds:.1f}s, ${r.estimated_cost_usd:.4f}")
        print(f"Last output: {r.last_output}")
        return None


if __name__ == "__main__":
    result = run_agent_with_watchdog("What is the capital of France?")
    print("Result:", result)
