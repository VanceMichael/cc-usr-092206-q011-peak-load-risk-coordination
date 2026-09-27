"""不可变观测存储。

- 只追加，不提供更新/删除接口：跨日峰谷等派生统计在读时计算，从不会改写原始观测。
- 计量更正以新观测链接（corrects）原观测；原始读数始终可取回。
- 迟到读数与更正在写入时给出受影响的时窗，供预测引擎做局部重算。
"""

from __future__ import annotations

from datetime import date

from .models import Observation, ObservationKind, Quality, Window


class ObservationStore:
    def __init__(self) -> None:
        self._by_id: dict[str, Observation] = {}
        self._order: list[str] = []
        self._corrections: dict[str, list[str]] = {}  # 被更正 id -> 更正观测 id 列表

    # ---- 写入（只追加） ----

    def append(self, obs: Observation) -> tuple[Observation, ...]:
        """写入一条观测，返回受影响的原始观测（迟到读数/更正波及的时窗来源）。"""
        if obs.id in self._by_id:
            raise ValueError(f"观测 id 重复: {obs.id}")
        if obs.corrects is not None and obs.corrects not in self._by_id:
            raise ValueError(f"更正目标不存在: {obs.corrects}")
        self._by_id[obs.id] = obs
        self._order.append(obs.id)
        affected = [obs]
        if obs.corrects is not None:
            self._corrections.setdefault(obs.corrects, []).append(obs.id)
            affected.append(self._by_id[obs.corrects])
        return tuple(affected)

    # ---- 读取 ----

    def get(self, obs_id: str) -> Observation:
        """取原始观测；即使已被更正，原读数也原样保留。"""
        return self._by_id[obs_id]

    def all_raw(self) -> tuple[Observation, ...]:
        return tuple(self._by_id[i] for i in self._order)

    def is_superseded(self, obs_id: str) -> bool:
        return obs_id in self._corrections

    def superseded_by(self, obs_id: str) -> tuple[str, ...]:
        return tuple(self._corrections.get(obs_id, ()))

    def effective_id(self, obs_id: str) -> str:
        """沿更正链取最新有效观测 id。"""
        seen = {obs_id}
        current = obs_id
        while current in self._corrections:
            nxt = self._corrections[current][-1]
            if nxt in seen:
                raise ValueError(f"更正链存在环: {obs_id}")
            seen.add(nxt)
            current = nxt
        return current

    def effective(self, obs_id: str) -> Observation:
        return self._by_id[self.effective_id(obs_id)]

    def effective_observations(
        self,
        kind: ObservationKind | None = None,
        region: str | None = None,
        window: Window | None = None,
        industry: str | None = None,
    ) -> tuple[Observation, ...]:
        """最新有效视图：被更正的原始观测不参与计算，但仍在库中保留。"""
        out: list[Observation] = []
        for oid in self._order:
            if self.is_superseded(oid):
                continue
            obs = self._by_id[oid]
            if kind is not None and obs.kind is not kind:
                continue
            if region is not None and obs.region != region:
                continue
            if window is not None and not obs.window.overlaps(window):
                continue
            if industry is not None and obs.industry != industry:
                continue
            out.append(obs)
        return tuple(out)

    def abnormal_observations(self) -> tuple[Observation, ...]:
        return tuple(
            self._by_id[i]
            for i in self._order
            if self._by_id[i].quality is not Quality.NORMAL
        )

    # ---- 派生统计（只读，不写回） ----

    def daily_peak_valley(
        self, region: str, day: date
    ) -> tuple[tuple[Window, float], tuple[Window, float]] | None:
        """跨日峰谷基于有效观测即时计算；原始观测保持原样。"""
        loads: list[tuple[Window, float]] = []
        for obs in self.effective_observations(kind=ObservationKind.METER, region=region):
            if obs.industry is not None:
                continue
            if obs.window.start.date() != day:
                continue
            if "load_mw" not in obs.metrics:
                continue
            loads.append((obs.window, obs.metrics["load_mw"]))
        if not loads:
            return None
        peak = max(loads, key=lambda item: item[1])
        valley = min(loads, key=lambda item: item[1])
        return peak, valley
