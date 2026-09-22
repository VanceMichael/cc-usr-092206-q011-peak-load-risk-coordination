"""不可篡改的观测台账。

设计约束（来自领域需求）：

* 原始观测只追加、永不改写。迟到读数照常追加；计量更正不覆盖旧值，
  而是追加一条 supersedes 旧观测的新读数，旧读数保留可审计。
* 异常读数可以打标记（人工或自动），标记同样只追加；标记后读数不再
  进入预测和预警，但仍留在台账里，供事后复盘定位误报。
* 台账是 SeriesView 的唯一事实来源；跨日峰谷之类的派生量一律从
  SeriesView 现算，绝不回写原始观测。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterable

from . import timewin

OBS_KINDS = (
    "regional_load",        # 地区实测计量负荷（MW）
    "planned_load",         # 规划/典型日基准曲线（MW，日下发，预测基准用）
    "industry_load",        # 行业计量负荷（MW，按 data_center / ev_charge / high_end_manufacturing / residential / other）
    "enterprise_load",      # 参与企业关口计量（MW，sector 存 enterprise_id）
    "weather",              # 气象观测（temp_c 气温，rainfall_mm 降雨，wind_km_h 风速）
    "maintenance",          # 设备检修（derated_mw 检修导致的供电能力扣减）
    "ev_demand",            # 充换电需求（MW，含预约）
)


class ObsFlag(str, Enum):
    OK = "ok"
    SUSPECT = "suspect"          # 存疑：进入计算但附带说明
    ANOMALOUS = "anomalous"      # 异常：不进入计算，复盘可见


@dataclass(frozen=True)
class Observation:
    obs_id: str
    kind: str
    window: datetime
    value: float
    region: str
    received_at: datetime
    sector: str | None = None          # industry_load 使用
    unit: str = "MW"
    supersedes: str | None = None      # 计量更正：指向被替代的 obs_id
    flags: tuple[str, ...] = ()        # 人工标记历史（只追加）
    metrics: dict[str, float] = field(default_factory=dict)  # 复合观测（气象：temp_c/rainfall_mm/wind_km_h）
    note: str = ""

    @property
    def window_key(self) -> str:
        return timewin.key(self.window)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["window"] = self.window.isoformat()
        d["received_at"] = self.received_at.isoformat()
        d["flags"] = list(self.flags)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Observation":
        return Observation(
            obs_id=d["obs_id"],
            kind=d["kind"],
            window=timewin.window_start(d["window"]),
            value=float(d["value"]),
            region=d["region"],
            received_at=datetime.fromisoformat(d["received_at"]),
            sector=d.get("sector"),
            unit=d.get("unit", "MW"),
            supersedes=d.get("supersedes"),
            flags=tuple(d.get("flags", ())),
            metrics=dict(d.get("metrics", {})),
            note=d.get("note", ""),
        )


@dataclass
class FlagRecord:
    """针对某条观测的标记动作，本身也是只追加的审计记录。"""
    obs_id: str
    flag: str
    marked_at: datetime
    reason: str = ""


class Ledger:
    """追加式观测台账。"""

    def __init__(self):
        self._obs: dict[str, Observation] = {}
        self._order: list[str] = []
        self._flag_log: list[FlagRecord] = []

    # ---- 写入 -----------------------------------------------------------

    def record(
        self,
        obs_id: str,
        kind: str,
        window: str | datetime,
        value: float,
        region: str,
        received_at: str | datetime | None = None,
        sector: str | None = None,
        unit: str = "MW",
        metrics: dict[str, float] | None = None,
        supersedes: str | None = None,
        note: str = "",
    ) -> Observation:
        if kind not in OBS_KINDS:
            raise ValueError(f"未知观测类型: {kind}")
        if obs_id in self._obs:
            raise ValueError(f"观测编号重复: {obs_id}")
        obs = Observation(
            obs_id=obs_id,
            kind=kind,
            window=timewin.window_start(window),
            value=float(value),
            region=region,
            received_at=datetime.fromisoformat(received_at) if isinstance(received_at, str)
            else (received_at or datetime.now()),
            sector=sector,
            unit=unit,
            metrics=dict(metrics or {}),
            supersedes=supersedes,
            note=note,
        )
        self._obs[obs_id] = obs
        self._order.append(obs_id)
        return obs

    def correct(self, new_id: str, old_id: str, new_value: float,
                received_at: str | datetime, note: str = "") -> Observation:
        """计量更正：追加一条替代读数，旧读数原样保留。"""
        if old_id not in self._obs:
            raise ValueError(f"被更正的观测不存在: {old_id}")
        old = self._obs[old_id]
        return self.record(
            new_id, old.kind, old.window, new_value, old.region,
            received_at=received_at, sector=old.sector, unit=old.unit,
            metrics=dict(old.metrics),
            supersedes=old_id, note=note or "计量更正",
        )

    def flag(self, obs_id: str, flag: ObsFlag, marked_at: str | datetime | None = None,
             reason: str = "") -> None:
        if obs_id not in self._obs:
            raise ValueError(f"观测不存在: {obs_id}")
        when = (datetime.fromisoformat(marked_at) if isinstance(marked_at, str)
                else (marked_at or datetime.now()))
        old = self._obs[obs_id]
        # 观测不可变：用新对象携带追加后的标记元组替换索引项（台账事件本身仍可审计）。
        self._obs[obs_id] = Observation(
            obs_id=old.obs_id, kind=old.kind, window=old.window, value=old.value,
            region=old.region, received_at=old.received_at, sector=old.sector,
            unit=old.unit, supersedes=old.supersedes,
            flags=old.flags + (flag.value,), metrics=dict(old.metrics),
            note=old.note,
        )
        self._flag_log.append(FlagRecord(obs_id, flag.value, when, reason))

    # ---- 读取 -----------------------------------------------------------

    def get(self, obs_id: str) -> Observation:
        return self._obs[obs_id]

    def all_observations(self) -> list[Observation]:
        return [self._obs[i] for i in self._order]

    def flag_log(self) -> list[FlagRecord]:
        return list(self._flag_log)

    def view(self, as_of: datetime | None = None) -> "SeriesView":
        return SeriesView(self, default_as_of=as_of)

    # ---- 持久化（整账导出/恢复；恢复时重建索引） ------------------------

    def save(self, path: str | Path) -> None:
        payload = {
            "observations": [self._obs[i].to_dict() for i in self._order],
            "flag_log": [
                {"obs_id": f.obs_id, "flag": f.flag,
                 "marked_at": f.marked_at.isoformat(), "reason": f.reason}
                for f in self._flag_log
            ],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        led = cls()
        for d in payload["observations"]:
            obs = Observation.from_dict(d)
            led._obs[obs.obs_id] = obs
            led._order.append(obs.obs_id)
        for f in payload.get("flag_log", []):
            led._flag_log.append(FlagRecord(
                f["obs_id"], f["flag"], datetime.fromisoformat(f["marked_at"]),
                f.get("reason", "")))
        return led


class SeriesView:
    """台账的派生只读视图：解析更正链、过滤被替代/异常读数。

    任何模型、预警只能通过本视图取数；跨日峰谷等派生量也从这里现算。
    """

    def __init__(self, ledger: Ledger, default_as_of: datetime | None = None):
        self._ledger = ledger
        self._default_as_of = default_as_of

    def effective(self, kinds: Iterable[str] | None = None,
                  include_suspect: bool = True,
                  as_of: datetime | None = -1) -> list[Observation]:
        """返回当前有效观测：未被替代、未被标记为 anomalous。

        as_of 给定后只含接收时间不晚于该时刻的观测，保证早时窗生成的
        预测版本“看不到未来才到的读数”，版本可严格复现。
        缺省使用视图的 default_as_of（如 led.view(as_of=...)）。
        """
        if as_of == -1:
            as_of = self._default_as_of
        kinds = set(kinds) if kinds else None
        # as_of 时点之后才到达的更正，在该时点尚不存在，不能压制原读数
        superseded_ids = {
            o.supersedes for o in self._ledger.all_observations()
            if o.supersedes and (as_of is None or o.received_at <= as_of)
        }
        out: list[Observation] = []
        # 标记也是只追加事件：as_of 时点只认当时已经打上的标记
        anom_ids, susp_ids = set(), set()
        for f in self._ledger.flag_log():
            if as_of is not None and f.marked_at > as_of:
                continue
            if f.flag == ObsFlag.ANOMALOUS.value:
                anom_ids.add(f.obs_id)
            elif f.flag == ObsFlag.SUSPECT.value:
                susp_ids.add(f.obs_id)
        if as_of is None:
            # 无时间截止时，直接用观测上累积的当前标记
            anom_ids = {o.obs_id for o in self._ledger.all_observations()
                        if ObsFlag.ANOMALOUS.value in o.flags}
            susp_ids = {o.obs_id for o in self._ledger.all_observations()
                        if ObsFlag.SUSPECT.value in o.flags}
        for o in self._ledger.all_observations():
            if o.obs_id in superseded_ids:
                continue
            if o.obs_id in anom_ids:
                continue
            if not include_suspect and o.obs_id in susp_ids:
                continue
            if as_of is not None and o.received_at > as_of:
                continue
            if kinds and o.kind not in kinds:
                continue
            out.append(o)
        return out

    def series(self, kind: str, region: str | None = None,
               sector: str | None = None,
               as_of: datetime | None = None) -> dict[datetime, Observation]:
        """按时窗起点组织的最新有效读数（一个时窗一条）。"""
        as_of = self._default_as_of if as_of is None else as_of
        out: dict[datetime, Observation] = {}
        for o in self.effective([kind], as_of=as_of):
            if region is not None and o.region != region:
                continue
            if sector is not None and o.sector != sector:
                continue
            # 更正后的新观测 received_at 更晚；同时存在多条时取最新接收。
            prev = out.get(o.window)
            if prev is None or o.received_at > prev.received_at:
                out[o.window] = o
        return out

    def at(self, kind: str, window: datetime, region: str | None = None,
           sector: str | None = None,
           as_of: datetime | None = None) -> Observation | None:
        return self.series(kind, region, sector, as_of).get(
            timewin.floor_to_window(window))

    def regions(self) -> list[str]:
        return sorted({o.region for o in self.effective()})

    def daily_extremes(self, kind: str, region: str, day: datetime,
                       as_of: datetime | None = None) -> dict:
        """跨日峰谷为派生视图：仅从有效观测现算，绝不回写台账。

        返回 {'max': (window, obs), 'min': (window, obs)}，可能为空。
        """
        day_obs = [o for w, o in self.series(kind, region, as_of=as_of).items()
                   if timewin.same_day(w, day)]
        if not day_obs:
            return {"max": None, "min": None}
        hi = max(day_obs, key=lambda o: o.value)
        lo = min(day_obs, key=lambda o: o.value)
        return {"max": (hi.window, hi), "min": (lo.window, lo)}
