"""Tests for long-lived foreground promotion and refusal diagnostics in terminal_tool.

Incident 2026-09-18: a local model emitted ``python3 -m http.server`` five times in a row as
``{"command": ...}`` with no ``background`` key, while its visible text claimed it had set the
flag. Each call hit the foreground long-lived guard, whose message said only "Run it with
background=true". That is unfalsifiable to a caller that cannot see its own emitted JSON, so it
retried identically until ``identical_call_streak_halt`` killed the run and the finished artifact
was never handed over.

Two fixes, both asserted here:

1. A plain server/watch invocation needs no command rewrite, so it is promoted to a tracked
   background session instead of refused, using the same reasoning the over-cap-timeout path
   already applies (refusals there only bought mechanical retries).
2. The remaining refusals (``&``, nohup/disown/setsid) state what the call actually arrived with,
   so one round trip corrects the caller instead of five.
"""
import json
import socket
from unittest.mock import patch

import pytest

from tools.terminal_tool import (
    _Rejected, _foreground_background_guidance, _foreground_background_verdict, _plan_execution,
    terminal_tool,
)
from tools.terminal_tool_guards import (
    GUIDANCE_AMP_BG, GUIDANCE_LONG_LIVED, GUIDANCE_SHELL_BG,
)


# The exact command from the incident (the 5th and final identical call).
INCIDENT_COMMAND = (
    "cd /home/al/coding/zfold-exploded && exec python3 -m http.server 8090 --bind 0.0.0.0"
)


def _make_env_config(**overrides):
    """Minimal _get_env_config()-shaped dict (mirrors the timeout-cap test helper)."""
    config = {
        "env_type": "local",
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    config.update(overrides)
    return config


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestVerdictClassification:
    """The guard must distinguish "needs a rewritten command" from "just needs promoting"."""

    def test_incident_command_is_long_lived(self):
        kind, msg = _foreground_background_verdict(INCIDENT_COMMAND)
        assert kind == GUIDANCE_LONG_LIVED
        assert "did not set" in msg

    def test_ampersand_and_nohup_stay_refusals(self):
        assert _foreground_background_verdict("sleep 5 &")[0] == GUIDANCE_AMP_BG
        assert _foreground_background_verdict("nohup make test")[0] == GUIDANCE_SHELL_BG

    def test_plain_command_unaffected(self):
        assert _foreground_background_verdict("echo hello") is None

    def test_help_variant_still_runs_in_foreground(self):
        assert _foreground_background_verdict("pnpm dev --help") is None


class TestRefusalNamesWhatArrived:
    """Fix 2: a refusal must say the call arrived without the flag, not just what to send."""

    def test_amp_refusal_states_the_missing_key(self):
        msg = _foreground_background_guidance("python3 server.py &")
        assert 'arrived without "background": true' in msg
        assert "'&' backgrounding" in msg
        assert "WITHOUT the '&'" in msg

    def test_nohup_refusal_states_the_missing_key(self):
        msg = _foreground_background_guidance("nohup ./worker.sh > /dev/null 2>&1")
        assert 'arrived without "background": true' in msg
        assert "WITHOUT the wrapper" in msg
        assert "notify_on_complete=true" in msg


class TestLongLivedPromotion:
    """Fix 1: the server pattern is promoted instead of refused, so it cannot loop."""

    def test_plan_marks_the_promotion_without_raising(self):
        plan = _plan_execution(
            INCIDENT_COMMAND, task_id=None, timeout=None, background=False, _host_local=True,
        )
        assert plan.promoted_from_long_lived_foreground is True
        assert plan.promoted_from_foreground_timeout is None

    def test_ampersand_and_nohup_plans_still_raise(self):
        for cmd in ("sleep 5 &", "nohup make test"):
            with pytest.raises(_Rejected):
                _plan_execution(cmd, task_id=None, timeout=None, background=False, _host_local=True)

    def test_explicit_background_flag_is_untouched(self):
        """A caller that already asked for background must not be double-promoted."""
        plan = _plan_execution(
            INCIDENT_COMMAND, task_id=None, timeout=None, background=True, _host_local=True,
        )
        assert plan.promoted_from_long_lived_foreground is False

    def test_incident_command_ends_in_a_background_session_not_a_refusal(self, tmp_path, monkeypatch):
        """Real local backend: the incident command becomes a tracked session, twice over."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
        cmd = f"timeout 3 python3 -m http.server {_free_port()} --bind 127.0.0.1"
        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config(cwd=str(tmp_path))), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
            first = json.loads(terminal_tool(command=cmd))
            second = json.loads(terminal_tool(command=cmd))

        for result in (first, second):
            assert result.get("error") is None, result
            assert result["output"] == "Background process started"
            assert result["session_id"].startswith("proc_")
            # A server never exits, so no completion notification may be promised.
            assert result.get("notify_on_complete") is not True
            note = result["promoted_from_foreground"]
            assert "tracked background session" in note
            assert "Do NOT re-run it." in note
            assert "process(action=\"kill\"" in note

        # Two identical calls produce two sessions: the identical-call streak that ended the
        # incident run can no longer accumulate from this command.
        assert first["session_id"] != second["session_id"]
