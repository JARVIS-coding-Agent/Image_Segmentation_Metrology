"""標準輸出訊息與單張失敗隔離工具（spec FR-015、contracts/cli.md）。"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from collections.abc import Iterator


def info(msg: str) -> None:
    print(f"[info] {msg}", file=sys.stderr)


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def skip(msg: str) -> None:
    print(f"[skip] {msg}", file=sys.stderr)


def error(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)


@contextmanager
def isolate_failure(image_id: object) -> Iterator[None]:
    """隔離單張圖片處理錯誤，記錄後續跑下一張，不中斷整批（spec SC-001）。"""
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - 刻意攔截以隔離單張失敗
        error(f"image={image_id} failed: {type(exc).__name__}: {exc}")
