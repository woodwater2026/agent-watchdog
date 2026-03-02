"""Tests for AgentWatchdog."""
import pytest
import time
from agent_watchdog import AgentWatchdog, WatchdogHalt, HaltReason


def test_loop_detection():
    wd = AgentWatchdog(max_identical_calls=3)
    with pytest.raises(WatchdogHalt) as exc_info:
        with wd.watch("test-loop"):
            wd.record_tool_call("search", args="query")
            wd.record_tool_call("search", args="query")
            wd.record_tool_call("search", args="query")  # triggers halt
    assert exc_info.value.report.reason == HaltReason.LOOP_DETECTED
    assert exc_info.value.report.run_id == "test-loop"


def test_no_loop_with_different_args():
    wd = AgentWatchdog(max_identical_calls=3)
    with wd.watch("test-no-loop"):
        wd.record_tool_call("search", args="query1")
        wd.record_tool_call("search", args="query2")
        wd.record_tool_call("search", args="query3")
    # Should complete without raising


def test_no_loop_with_different_tools():
    wd = AgentWatchdog(max_identical_calls=3)
    with wd.watch("test-mixed"):
        wd.record_tool_call("search", args="q")
        wd.record_tool_call("read_file", args="q")
        wd.record_tool_call("search", args="q")
    # Different tools in between, no loop


def test_budget_exceeded():
    wd = AgentWatchdog(max_budget_usd=0.001, model="anthropic/claude-sonnet-4-6")
    with pytest.raises(WatchdogHalt) as exc_info:
        with wd.watch("test-budget"):
            wd.record_tokens(token_in=1000, token_out=100)  # ~$0.0045 > $0.001
    assert exc_info.value.report.reason == HaltReason.BUDGET_EXCEEDED


def test_budget_ok():
    wd = AgentWatchdog(max_budget_usd=10.0, model="anthropic/claude-sonnet-4-6")
    with wd.watch("test-budget-ok"):
        wd.record_tokens(token_in=100, token_out=10)
    # Should not raise


def test_timeout():
    wd = AgentWatchdog(timeout_seconds=0.2)
    with pytest.raises(WatchdogHalt) as exc_info:
        with wd.watch("test-timeout"):
            time.sleep(0.5)  # exceeds 0.2s timeout
    assert exc_info.value.report.reason == HaltReason.TIMEOUT


def test_halt_report_structure():
    wd = AgentWatchdog(max_identical_calls=2)
    with pytest.raises(WatchdogHalt) as exc_info:
        with wd.watch("test-report"):
            wd.record_tool_call("search", args="q", output="some result")
            wd.record_tool_call("search", args="q")
    report = exc_info.value.report
    assert report.run_id == "test-report"
    assert report.elapsed_seconds >= 0
    assert len(report.tool_calls) == 2
    assert report.last_output == "some result"


def test_manual_halt():
    wd = AgentWatchdog()
    with pytest.raises(WatchdogHalt) as exc_info:
        with wd.watch("test-manual"):
            wd.halt("user requested stop")
    assert exc_info.value.report.reason == HaltReason.MANUAL


def test_no_watch_context():
    """record_* calls outside watch() should be no-ops."""
    wd = AgentWatchdog()
    wd.record_tool_call("anything", args="x")  # should not raise
    wd.record_tokens(1000, 500)  # should not raise
