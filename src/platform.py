"""日内值班编排：持续接收数据，自动做增量预测、预警与需求响应对账。

Platform 不替代各模块，只负责把它们按值班节奏串起来：

* ingest：计量/气象/检修/充换电观测进入台账（只追加）；
* 迟到读数、计量更正后调用 refresh：只重算受影响时窗，生成新版本；
* 新版本预警批次与现有有效指令对账：消失的预警撤回指令（容量恢复），
  仍存在或新增的缺口在资源上限内补充分配；
* 企业通过 enterprise_view 只看到自己的行动要求；
* 分析人员用 explain_alert 逐项查看数据与假设；
* 事后用 Reviewer 对照实际削峰、未履约与异常误报。
"""

from __future__ import annotations

from datetime import datetime

from . import timewin
from .alerts import AlertEngine, AlertBatch
from .forecast import Forecaster, ForecastVersion, ForecastAssumptions
from .ledger import Ledger
from .response import ResponseCoordinator
from .review import Reviewer, ReviewReport


class Platform:
    def __init__(self, capacity_mw: dict[str, float], regions: list[str],
                 horizon: list[datetime],
                 assumptions: ForecastAssumptions | None = None,
                 policy=None):
        self.ledger = Ledger()
        self.coordinator = ResponseCoordinator()
        self.forecaster = Forecaster(assumptions)
        self.alert_engine = AlertEngine(capacity_mw, policy)
        self.regions = list(regions)
        self.horizon = list(horizon)
        self.versions: list[ForecastVersion] = []
        self.batches: list[AlertBatch] = []
        self._seq = 0

    # ---- 数据接入 -------------------------------------------------------

    def ingest(self, obs_id: str, kind: str, window, value: float,
               region: str, received_at, **kwargs):
        return self.ledger.record(obs_id, kind, window, value, region,
                                  received_at=received_at, **kwargs)

    def correct(self, new_id: str, old_id: str, value: float, received_at,
                reason: str = "计量更正"):
        return self.ledger.correct(new_id, old_id, value, received_at, reason)

    # ---- 预测 / 预警 / 响应 一轮闭环 -----------------------------------

    def refresh(self, now: datetime, trigger: str = "manual",
                changed_obs: list[str] | None = None,
                static_baseline: dict | None = None,
                note: str = "") -> tuple[ForecastVersion, AlertBatch]:
        """生成新预测版本与预警批次，并完成指令对账与补充分配。

        changed_obs 为空时重算全部时窗（如例行调度）；否则只重算受影响窗。
        """
        parent = self.versions[-1] if self.versions else None
        if changed_obs:
            affected = self.forecaster.affected_windows_for(
                changed_obs, self.ledger.view(), self.horizon)
        else:
            affected = [timewin.key(w) for w in self.horizon]
        self._seq += 1
        vid = f"FV-{now:%Y%m%d}-{self._seq:02d}"
        fv = self.forecaster.run(
            vid, self.ledger.view(), self.horizon, self.regions, now,
            trigger=trigger, affected_windows=affected, parent=parent,
            static_baseline=static_baseline or {}, note=note)
        batch = self.alert_engine.evaluate(
            f"AB-{self._seq:02d}", fv, self.ledger.view(), self.horizon, now)
        # 与既有指令对账：预警不再覆盖的时窗撤回，容量恢复
        active_keys = {(a.region, a.window) for a in batch.alerts if a.status == "active"}
        self.coordinator.reconcile(active_keys, now)
        # 对仍存在/新增的缺口补充分配（已覆盖部分不重复）
        self.coordinator.allocate_for_alerts(batch.alerts, now)
        self.versions.append(fv)
        self.batches.append(batch)
        return fv, batch

    # ---- 对外视图 -------------------------------------------------------

    def enterprise_view(self, enterprise_id: str):
        return self.coordinator.enterprise_view(enterprise_id)

    def explain_alert(self, alert_id: str):
        for b in reversed(self.batches):
            for a in b.alerts:
                if a.alert_id == alert_id:
                    return {
                        "alert": a.to_dict(),
                        "explanation": a.explanation,
                        "forecast_version": a.forecast_version,
                    }
        raise ValueError(f"预警不存在: {alert_id}")

    def review(self, report_id: str, batch: AlertBatch, fv: ForecastVersion,
               windows: list[datetime], now: datetime) -> ReviewReport:
        return Reviewer(self.ledger, self.coordinator,
                        self.alert_engine.capacity).run(
            report_id, batch, fv, windows, now)
