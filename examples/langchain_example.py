"""
Agent Watchdog + LangChain integration example.

Shows how to wrap a LangChain agent with watchdog monitoring.
"""
from agent_watchdog import AgentWatchdog, WatchdogHalt

# --- Setup ---
# pip install agent-watchdog langchain langchain-openai

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool


@tool
def search(query: str) -> str:
    """Search the web for information."""
    # Replace with real search tool
    return f"Results for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# Build a standard LangChain agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [search, calculate]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=20)

# --- Watchdog wrapper ---

watchdog = AgentWatchdog(
    max_budget_usd=0.50,       # halt if run exceeds $0.50
    max_identical_calls=3,     # halt if same tool+args called 3x in a row
    timeout_seconds=60,        # halt after 60 seconds
    model="openai/gpt-4o",
)


def watched_invoke(task: str, run_id: str = "langchain-run") -> dict:
    """Run a LangChain agent with watchdog monitoring."""
    try:
        with watchdog.watch(run_id=run_id):
            # Patch: record tool calls via callbacks
            # For LangChain, use a custom callback handler
            result = executor.invoke({"input": task})
            return {"ok": True, "output": result["output"]}

    except WatchdogHalt as e:
        report = e.report
        return {
            "ok": False,
            "halted": True,
            "reason": report.reason.value,
            "cost_usd": report.estimated_cost_usd,
            "calls": len(report.tool_calls),
            "message": report.message,
        }


# --- With callback for per-step monitoring ---

from langchain_core.callbacks import BaseCallbackHandler


class WatchdogCallback(BaseCallbackHandler):
    """LangChain callback that feeds tool calls into AgentWatchdog."""

    def __init__(self, watchdog: AgentWatchdog, model: str = "openai/gpt-4o"):
        self.watchdog = watchdog
        self.model = model

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        self.watchdog.record_tool_call(tool_name, args=input_str)

    def on_tool_end(self, output, **kwargs):
        pass  # output recorded in on_tool_start via WatchdogHalt if loop

    def on_llm_end(self, response, **kwargs):
        # Approximate token counting from LangChain response
        usage = getattr(response, "llm_output", {}) or {}
        token_usage = usage.get("token_usage", {})
        self.watchdog.record_tokens(
            token_in=token_usage.get("prompt_tokens", 0),
            token_out=token_usage.get("completion_tokens", 0),
        )


def watched_invoke_with_callbacks(task: str, run_id: str = "langchain-run") -> dict:
    """Full integration: watchdog monitors every tool call and token."""
    callback = WatchdogCallback(watchdog, model="openai/gpt-4o-mini")

    try:
        with watchdog.watch(run_id=run_id):
            result = executor.invoke(
                {"input": task},
                config={"callbacks": [callback]},
            )
            return {"ok": True, "output": result["output"]}

    except WatchdogHalt as e:
        report = e.report
        return {
            "ok": False,
            "halted": True,
            "reason": report.reason.value,
            "cost_usd": report.estimated_cost_usd,
            "calls": len(report.tool_calls),
            "last_output": report.last_output,
            "message": report.message,
        }


# --- Demo ---
if __name__ == "__main__":
    result = watched_invoke_with_callbacks(
        "What is 2 + 2? Then search for the latest news on AI agents.",
        run_id="demo-001",
    )
    print(result)
