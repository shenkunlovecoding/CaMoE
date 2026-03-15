"""One-time logging for backend kernel initialization."""

from __future__ import annotations

_EMITTED: set[tuple[str, str]] = set()


def announce_kernel_init(family: str, backend: str, *, detail: str | None = None) -> None:
    key = (str(family), str(backend))
    if key in _EMITTED:
        return
    _EMITTED.add(key)
    suffix = f" | {detail}" if detail else ""
    print(f"[kernel-init] family={family} backend={backend}{suffix}")
