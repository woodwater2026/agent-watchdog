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


# --- Pattern detection tests ---

def test_pattern_detection_abab():
    """ABAB alternating pattern should be caught by sliding-window detection."""
    wd = AgentWatchdog(max_identical_calls=10, pattern_window_size=8)
    with pytest.raises(WatchdogHalt) as exc_info:
        with wd.watch("test-abab"):
            for i in range(10):
                tool = "search" if i % 2 == 0 else "read_page"
                wd.record_tool_call(tool, args="same_query")
    report = exc_info.value.report
    assert report.reason == HaltReason.LOOP_DETECTED
    assert "pattern" in report.message
    assert len(report.tool_calls) == 8  # caught exactly at window fill


def test_pattern_detection_abcabc():
    """ABCABC 3-cycle pattern should be caught."""
    wd = AgentWatchdog(max_identical_calls=10, pattern_window_size=6)
    with pytest.raises(WatchdogHalt) as exc_info:
        with wd.watch("test-abc"):
            for i in range(10):
                tool = ["search", "read", "write"][i % 3]
                wd.record_tool_call(tool, args="q")
    report = exc_info.value.report
    assert report.reason == HaltReason.LOOP_DETECTED
    assert len(report.tool_calls) == 6


def test_pattern_detection_disabled():
    """pattern_window_size=0 should disable pattern detection."""
    wd = AgentWatchdog(max_identical_calls=10, pattern_window_size=0, timeout_seconds=5)
    with wd.watch("test-no-pattern"):
        for i in range(10):
            tool = "search" if i % 2 == 0 else "read_page"
            wd.record_tool_call(tool, args="same_query")
    # Should NOT raise — pattern detection is disabled


def test_no_false_positive_varied_calls():
    """Non-repeating sequence should never trip pattern detection."""
    wd = AgentWatchdog(max_identical_calls=10, pattern_window_size=8)
    with wd.watch("test-varied"):
        for i in range(8):
            wd.record_tool_call(f"tool_{i}", args=f"unique_{i}")
    # Should complete cleanly


def test_detect_repeating_pattern_helper():
    """Unit test for _detect_repeating_pattern directly."""
    from agent_watchdog.watchdog import _detect_repeating_pattern

    assert _detect_repeating_pattern([("A", "h"), ("B", "h"), ("A", "h"), ("B", "h")]) == [("A", "h"), ("B", "h")]
    assert _detect_repeating_pattern([("A", "1"), ("B", "2"), ("C", "3")] * 2) == [("A", "1"), ("B", "2"), ("C", "3")]
    # AAAA: shortest repeating unit at pat_len=2 is [A,A]; pat_len=1 not checked
    # (single-identical handled by max_identical_calls, not pattern detection)
    assert _detect_repeating_pattern([("A", "h")] * 4) == [("A", "h"), ("A", "h")]
    assert _detect_repeating_pattern([("A", "1"), ("B", "2"), ("C", "3"), ("D", "4")]) == []
