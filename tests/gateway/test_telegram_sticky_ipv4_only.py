"""Sticky path may only ever be an IPv4 literal (2026-11-12, DSA F1).

A hostname success used to set the sticky slot to None ("sticky hostname"), and
``TelegramFallbackTransport._attempt_order()`` puts the sticky entry FIRST — so the
adapter actively preferred the dual-stack hostname after any single hostname success,
which is the blackholed-AAAA path #87015 exists to avoid. gateway.log showed hostname
stickies in 3 of 5 sampled flap lines.

Runs standalone (no pytest in this env):  python3 venv/bin/... -m tests.gateway...
    cd ~/.hermes/hermes-agent && venv/bin/python3 -m tests.gateway.test_telegram_sticky_ipv4_only
Plain-assert functions named ``test_*`` are still collected if pytest is ever installed.
"""

from __future__ import annotations

import asyncio

import httpx

from plugins.platforms.telegram import telegram_network as tnet


class _Fake:
    """Minimal transport: succeed, or raise a retryable connect error."""

    def __init__(self, ok: bool):
        self.ok = ok

    async def handle_async_request(self, request):
        if not self.ok:
            raise httpx.ConnectError("fake connect failure", request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    async def aclose(self):
        return None


def _transport(fallback_ips):
    return tnet.TelegramFallbackTransport(fallback_ips)


def _wire(t, ok_for):
    """Point the transport at fakes. ``ok_for`` maps None -> hostname, else the IP."""
    t._primary = _Fake(ok_for.get(None, False))

    async def _get_fallback(ip):
        return _Fake(ok_for.get(ip, False))

    async def _noop(*_a, **_kw):
        return None

    t._get_fallback = _get_fallback
    t._reset_primary = _noop
    t._reset_fallback = _noop


def _request():
    return httpx.Request("GET", "https://api.telegram.org/bot123/getMe")


def test_ipv4_success_becomes_sticky_and_goes_first():
    t = _transport(["149.154.166.110", "149.154.167.220"])
    _wire(t, {None: False, "149.154.166.110": False, "149.154.167.220": True})

    asyncio.run(t.handle_async_request(_request()))

    assert t._sticky_ip == "149.154.167.220", t._sticky_ip
    assert t._attempt_order()[0] == "149.154.167.220", t._attempt_order()


def test_hostname_success_never_becomes_sticky():
    t = _transport(["149.154.166.110", "149.154.167.220"])
    _wire(t, {None: True, "149.154.166.110": False, "149.154.167.220": False})

    asyncio.run(t.handle_async_request(_request()))

    # Unset, not None: "sticky hostname" is the bug this test exists to prevent.
    assert t._sticky_ip is tnet._UNSET, t._sticky_ip
    order = t._attempt_order()
    assert order[:2] == ["149.154.166.110", "149.154.167.220"], order
    assert order[-1] is None, order  # hostname stays last-resort


def test_failed_sticky_falls_back_to_the_literal_walk():
    t = _transport(["149.154.166.110", "149.154.167.220"])
    _wire(t, {None: False, "149.154.166.110": False, "149.154.167.220": True})
    asyncio.run(t.handle_async_request(_request()))
    assert t._sticky_ip == "149.154.167.220", t._sticky_ip

    # The pinned IP now dies: sticky must clear and the walk must retry the others.
    _wire(t, {None: False, "149.154.166.110": True, "149.154.167.220": False})
    asyncio.run(t.handle_async_request(_request()))

    assert t._sticky_ip == "149.154.166.110", t._sticky_ip
    assert t._attempt_order()[0] == "149.154.166.110", t._attempt_order()


def test_all_paths_failing_raises_the_last_connect_error():
    t = _transport(["149.154.166.110"])
    _wire(t, {None: False, "149.154.166.110": False})

    try:
        asyncio.run(t.handle_async_request(_request()))
    except httpx.ConnectError:
        return
    raise AssertionError("expected the last ConnectError to propagate")


def main() -> int:
    cases = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in cases:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
        else:
            print(f"PASS  {name}")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
