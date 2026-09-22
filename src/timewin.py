"""统一时窗：平台内部一律使用 15 分钟时窗，时窗起点为右开边界 [start, start+15min)。"""

from __future__ import annotations

from datetime import datetime, timedelta

STEP = timedelta(minutes=15)


def floor_to_window(ts: datetime) -> datetime:
    """把任意时间戳归入其所属时窗的起点（整刻钟）。"""
    minute = (ts.minute // 15) * 15
    return ts.replace(minute=minute, second=0, microsecond=0)


def window_start(value: str | datetime) -> datetime:
    """接受 ISO 字符串或 datetime，返回时窗起点。"""
    ts = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    return floor_to_window(ts)


def key(ts: datetime) -> str:
    """时窗键，便于做字典键与持久化。"""
    return floor_to_window(ts).isoformat()


def enumerate_windows(start: datetime, end_exclusive: datetime):
    """按 15 分钟枚举时窗起点。"""
    cur = floor_to_window(start)
    end = floor_to_window(end_exclusive)
    while cur < end:
        yield cur
        cur += STEP


def same_day(a: datetime, b: datetime) -> bool:
    return a.date() == b.date()
