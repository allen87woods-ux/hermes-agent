"""Reload a gateway that is running stale checkout code.

``gateway/code_skew.py`` detects drift (boot fingerprint vs the checkout on disk)
but only two narrow callers used it (``/model`` switching and the dashboard skew
guard) and neither recovers.  Every other turn on a long-lived gateway therefore
keeps the modules imported at boot, so a commit to the checkout does not reach
Telegram, cron or the API server until somebody restarts the process by hand.

This module closes that hole.  A daemon thread checks the fingerprint on an
interval and, when the checkout has moved, requests the gateway's OWN graceful
restart (``GatewayRunner.request_restart`` -> draining stop -> exit 75 -> systemd
respawn, the same path ``/restart`` uses) as soon as no work is in flight.

Safety properties:

* Nothing is ever interrupted.  The request is only made when
  ``_active_work_count()`` reads zero, and an unreadable count is treated as
  work: never restart blind.
* ``request_restart`` is called on the event loop (``call_soon_threadsafe``) --
  it creates an asyncio task, which fails from a bare thread.
* A restart already in progress makes the request a no-op (``request_restart``
  returns False the second time).
* One request per detected drift, re-armed after ``_REARM_AFTER_SECONDS``, so a
  request that never landed cannot leave the gateway permanently stale.
* Disables itself when there is no boot fingerprint (non-git install): drift is
  unmeasurable there, so it must not guess.

Detection is cheap: the fingerprint is a file read (``hermes_cli.build_info``
deliberately avoids spawning git), and a clean checkout costs one string compare
per interval.
"""

from __future__ import annotations

import functools
import logging
import threading
import time

logger = logging.getLogger(__name__)

CHECK_INTERVAL_S = 30.0
_REARM_AFTER_SECONDS = 300.0

_lock = threading.Lock()
_armed_at: float | None = None
_watcher: threading.Thread | None = None
_waiter: threading.Thread | None = None


def _detect() -> tuple[str, str] | None:
    """``(boot_rev, disk_rev)`` when the checkout drifted, else ``None``."""
    try:
        from gateway.code_skew import detect_code_skew

        return detect_code_skew()
    except Exception:
        logger.debug("code-skew check unavailable", exc_info=True)
        return None


def _work_count(runner) -> int | None:
    """In-flight agent work, or ``None`` when it cannot be read."""
    try:
        return int(runner._active_work_count())
    except Exception:
        logger.debug("code-skew: active work count unreadable", exc_info=True)
        return None


def _request(runner, boot_rev: str, disk_rev: str) -> bool:
    """Ask the runner for its graceful restart.  Loop thread only. Never raises."""
    try:
        ok = bool(runner.request_restart(detached=False, via_service=True))
    except Exception:
        logger.exception("code-skew reload: restart request failed")
        return False
    if ok:
        message = (
            f"code-skew reload: gateway booted on {boot_rev} but the checkout is "
            f"{disk_rev}; requesting a graceful restart to load current code"
        )
        logger.warning(message)
        print(f"[ok] code-skew reload: {boot_rev} -> {disk_rev}; graceful restart requested", flush=True)
    else:
        logger.info("code-skew reload: a restart is already in progress; nothing to do")
    return ok


def _dispatch_default(fn) -> None:
    fn()


def _wait_then_request(runner, *, submit, poll_seconds: float) -> None:
    """Wait for in-flight work to drain, then ask for the restart exactly once."""
    while True:
        time.sleep(max(0.05, float(poll_seconds)))
        skew = _detect()
        if not skew:
            # Someone else reloaded, or the checkout moved back: nothing to do.
            return
        count = _work_count(runner)
        if count is None:
            continue  # unknown: keep waiting, never restart blind
        if count == 0:
            submit(functools.partial(_request, runner, skew[0], skew[1]))
            return


def _start_waiter(runner, *, submit, poll_seconds: float) -> None:
    global _waiter
    with _lock:
        if _waiter is not None and _waiter.is_alive():
            return
        thread = threading.Thread(
            target=_wait_then_request,
            args=(runner,),
            kwargs={"submit": submit, "poll_seconds": poll_seconds},
            name="code-skew-waiter",
            daemon=True,
        )
        _waiter = thread
    thread.start()


def request_reload_if_stale(
    runner,
    *,
    now: float | None = None,
    submit=None,
    poll_seconds: float | None = None,
) -> str:
    """Check for checkout drift and, if idle, request the graceful reload.

    Returns a state string for logging and tests:
    ``clean`` | ``queued`` | ``restart-requested`` | ``already-queued`` |
    ``unreadable`` | ``dispatch-failed``.
    """
    skew = _detect()
    if not skew:
        return "clean"
    boot_rev, disk_rev = skew
    moment = time.monotonic() if now is None else now

    global _armed_at
    with _lock:
        if _armed_at is not None and moment - _armed_at < _REARM_AFTER_SECONDS:
            return "already-queued"
        _armed_at = moment

    dispatch = submit or _dispatch_default
    poll = CHECK_INTERVAL_S if poll_seconds is None else poll_seconds

    count = _work_count(runner)
    if count is None:
        print(
            "[!!] code-skew reload: cannot read the active work count; not restarting",
            flush=True,
        )
        return "unreadable"
    if count > 0:
        print(
            f"[!!] code-skew reload: running {boot_rev}, checkout is {disk_rev}; "
            f"{count} work unit(s) in flight, reload queued until they finish",
            flush=True,
        )
        _start_waiter(runner, submit=dispatch, poll_seconds=poll)
        return "queued"

    try:
        dispatch(functools.partial(_request, runner, boot_rev, disk_rev))
    except Exception:
        logger.exception("code-skew reload: could not dispatch the restart request")
        return "dispatch-failed"
    return "restart-requested"


def _watch(runner, interval_s: float, submit) -> None:
    while True:
        time.sleep(max(5.0, float(interval_s)))
        try:
            state = request_reload_if_stale(runner, submit=submit)
        except Exception:
            logger.exception("code-skew reload: check raised")
            continue
        if state == "restart-requested":
            # The runner is draining; this process is about to be replaced.
            return


def start_code_skew_watcher(runner, *, interval_s: float = CHECK_INTERVAL_S) -> bool:
    """Start the periodic drift watcher.  Idempotent; never raises.

    ``submit`` (default: run inline) must be ``loop.call_soon_threadsafe`` when
    started from the event loop, because ``request_restart`` creates an asyncio
    task and cannot be called from this thread.
    """
    global _watcher
    try:
        submit = _resolve_submit()
        try:
            import gateway.code_skew as _code_skew

            if getattr(_code_skew, "_boot_fingerprint", None) is None:
                print(
                    "[!!] code-skew reload: no boot fingerprint (not a git checkout); "
                    "watcher disabled",
                    flush=True,
                )
                return False
        except Exception:
            print("[!!] code-skew reload: code_skew unavailable; watcher disabled", flush=True)
            return False

        with _lock:
            if _watcher is not None and _watcher.is_alive():
                return False
            thread = threading.Thread(
                target=_watch,
                args=(runner, float(interval_s), submit),
                name="code-skew-watcher",
                daemon=True,
            )
            _watcher = thread
        thread.start()
        print(
            "[ok] code-skew reload: watcher started "
            f"(checkout drift triggers a graceful restart when idle, every {float(interval_s):.0f}s)",
            flush=True,
        )
        return True
    except Exception as exc:  # never block gateway startup
        print(f"[!!] code-skew reload: watcher failed to start: {exc}", flush=True)
        return False


def _resolve_submit():
    """Best-effort ``call_soon_threadsafe`` for the running loop, else inline."""
    try:
        import asyncio

        loop = asyncio.get_running_loop()
    except Exception:
        return _dispatch_default

    def _submit(fn) -> None:
        loop.call_soon_threadsafe(fn)

    return _submit
