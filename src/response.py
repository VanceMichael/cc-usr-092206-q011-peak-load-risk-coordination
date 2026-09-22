"""需求响应资源、调度指令与容量恢复。

* 企业注册自己在各时窗可提供的削峰能力与基线负荷；
* 协调器依据预警缺口在地区内按确定性顺序（注册容量比例）分配；
* 所有指令只追加；企业退出响应或调度撤回指令时，通过状态事件把
  已分配容量即时恢复，可供同一时窗重新分配给其他企业；
* 企业视图严格按 enterprise_id 隔离——参与企业只能看到自己的行动要求；
* 窗口结束后用企业关口计量（enterprise_load）对照注册基线结算
  实际削峰与未履约容量。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from . import timewin
from .alerts import Alert
from .ledger import SeriesView


@dataclass
class Resource:
    """参与企业注册的可调度资源。"""
    enterprise_id: str
    name: str
    region: str
    # 时窗键 -> 可削减容量 MW
    offers: dict[str, float] = field(default_factory=dict)
    # 时窗键 -> 正常基线负荷 MW（结算用）
    baseline: dict[str, float] = field(default_factory=dict)

    def offer_at(self, window: str | datetime) -> float:
        return float(self.offers.get(timewin.key(
            window if isinstance(window, datetime)
            else datetime.fromisoformat(window)), 0.0))

    def baseline_at(self, window: str | datetime) -> float:
        wk = timewin.key(window if isinstance(window, datetime)
                         else datetime.fromisoformat(window))
        return float(self.baseline.get(wk, 0.0))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Resource":
        return Resource(**d)


@dataclass
class StatusEvent:
    at: datetime
    status: str
    actor: str                 # dispatcher / enterprise / system
    reason: str = ""

    def to_dict(self) -> dict:
        return {"at": self.at.isoformat(), "status": self.status,
                "actor": self.actor, "reason": self.reason}

    @staticmethod
    def from_dict(d: dict) -> "StatusEvent":
        return StatusEvent(datetime.fromisoformat(d["at"]), d["status"],
                           d["actor"], d.get("reason", ""))


ACTIVE_STATUSES = ("issued",)
TERMINAL_STATUSES = ("opt_out", "withdrawn", "settled")


@dataclass
class Directive:
    directive_id: str
    enterprise_id: str
    region: str
    window: str
    allocated_mw: float
    alert_id: str
    status: str = "issued"
    history: list[StatusEvent] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    def to_dict(self) -> dict:
        return {
            "directive_id": self.directive_id,
            "enterprise_id": self.enterprise_id,
            "region": self.region,
            "window": self.window,
            "allocated_mw": self.allocated_mw,
            "alert_id": self.alert_id,
            "status": self.status,
            "history": [h.to_dict() for h in self.history],
        }

    @staticmethod
    def from_dict(d: dict) -> "Directive":
        return Directive(
            directive_id=d["directive_id"],
            enterprise_id=d["enterprise_id"],
            region=d["region"],
            window=d["window"],
            allocated_mw=float(d["allocated_mw"]),
            alert_id=d["alert_id"],
            status=d["status"],
            history=[StatusEvent.from_dict(h) for h in d.get("history", [])],
        )


@dataclass
class WindowSettlement:
    enterprise_id: str
    window: str
    directive_id: str
    allocated_mw: float
    baseline_mw: float
    actual_mw: float
    actual_reduction_mw: float
    unperformed_mw: float

    def to_dict(self):
        return asdict(self)


class ResponseCoordinator:
    def __init__(self):
        self.resources: dict[str, Resource] = {}
        self.directives: list[Directive] = []
        # 企业退出或指令被撤回后，在该时窗不再参与重新分配，
        # 释放的容量流向其他企业；如需重新参与须显式 reallow。
        self._locked_out: dict[tuple[str, str], str] = {}

    # ---- 资源注册 -------------------------------------------------------

    def register(self, resource: Resource) -> None:
        self.resources[resource.enterprise_id] = resource

    # ---- 分配 -----------------------------------------------------------

    def available(self, enterprise_id: str, window: str | datetime) -> float:
        """企业在某时窗当前仍可调用的容量（注册量 - 有效指令占用）。"""
        res = self.resources[enterprise_id]
        wk = timewin.key(window if isinstance(window, datetime)
                         else datetime.fromisoformat(window))
        used = sum(d.allocated_mw for d in self.directives
                   if d.enterprise_id == enterprise_id
                   and d.window == wk and d.active)
        return round(res.offer_at(wk) - used, 3)

    def allocate_for_alerts(self, alerts: list[Alert],
                            created_at: datetime,
                            id_prefix: str = "D") -> list[Directive]:
        """按预警缺口分配；已由有效指令覆盖的缺口不重复分配。

        同一时窗同一地区内，缺口在各企业“剩余可分配容量”之间按比例分摊，
        确定性、可复算。
        """
        new_directives: list[Directive] = []
        seq = len(self.directives)
        # 地区+时窗聚合缺口，避免同一预警重复分配
        groups: dict[tuple[str, str], float] = {}
        alert_ref: dict[tuple[str, str], str] = {}
        for a in alerts:
            if a.status != "active":
                continue
            k = (a.region, a.window)
            groups[k] = groups.get(k, 0.0) + a.gap_mw
            alert_ref.setdefault(k, a.alert_id)
        for (region, wk), gap in sorted(groups.items()):
            remaining = round(gap - self._covered(region, wk), 3)
            if remaining <= 0:
                continue
            candidates = [
                (eid, self.available(eid, wk))
                for eid, r in sorted(self.resources.items())
                if r.region == region and self.available(eid, wk) > 0
                and (eid, wk) not in self._locked_out
            ]
            total_offer = sum(c for _, c in candidates)
            if total_offer <= 0:
                continue
            for eid, cap in candidates:
                share = round(min(cap, remaining * cap / total_offer), 3)
                if share <= 0:
                    continue
                seq += 1
                d = Directive(
                    directive_id=f"{id_prefix}{seq:04d}",
                    enterprise_id=eid, region=region, window=wk,
                    allocated_mw=share, alert_id=alert_ref[(region, wk)],
                    history=[StatusEvent(created_at, "issued", "dispatcher",
                                         "按预警缺口比例分配")])
                self.directives.append(d)
                new_directives.append(d)
        return new_directives

    def _covered(self, region: str, wk: str) -> float:
        return round(sum(d.allocated_mw for d in self.directives
                         if d.region == region and d.window == wk and d.active), 3)

    # ---- 退出 / 撤回：容量即时恢复 -------------------------------------

    def _transition(self, directive: Directive, status: str, actor: str,
                    at: datetime, reason: str) -> None:
        if not directive.active:
            raise ValueError(f"指令 {directive.directive_id} 已终态"
                             f"（{directive.status}），不能再变更")
        directive.status = status
        directive.history.append(StatusEvent(at, status, actor, reason))
        # 容量即时恢复（active 指令释放）。企业主动退出时锁定其在本时窗的
        # 再分配资格（须显式 reallow）；调度撤回只是解除本次调用，企业
        # 资源在预警重现时仍可被重新调用。
        if status == "opt_out":
            self._locked_out[(directive.enterprise_id,
                              directive.window)] = status

    def reallow(self, enterprise_id: str, window: str | datetime) -> None:
        """显式允许企业在退出/撤回后重新参与该时窗（如新的资源确认）。"""
        wk = timewin.key(window if isinstance(window, datetime)
                         else datetime.fromisoformat(window))
        self._locked_out.pop((enterprise_id, wk), None)

    def enterprise_opt_out(self, enterprise_id: str, at: datetime,
                           directive_ids: list[str] | None = None,
                           reason: str = "企业退出响应") -> list[Directive]:
        targets = [d for d in self.directives
                   if d.enterprise_id == enterprise_id and d.active
                   and (directive_ids is None or d.directive_id in directive_ids)]
        for d in targets:
            self._transition(d, "opt_out", enterprise_id, at, reason)
        return targets

    def withdraw(self, directive_id: str, at: datetime,
                 reason: str = "调度撤回指令") -> Directive:
        d = self._find(directive_id)
        self._transition(d, "withdrawn", "dispatcher", at, reason)
        return d

    def _find(self, directive_id: str) -> Directive:
        for d in self.directives:
            if d.directive_id == directive_id:
                return d
        raise ValueError(f"指令不存在: {directive_id}")

    # ---- 企业隔离视图 ---------------------------------------------------

    def enterprise_view(self, enterprise_id: str) -> list[dict]:
        """参与企业只收到自己的行动要求，看不到其他企业或预警全貌。"""
        return [{
            "directive_id": d.directive_id,
            "window": d.window,
            "region": d.region,
            "required_reduction_mw": d.allocated_mw,
            "status": d.status,
            "action": (f"请于 {d.window} 起在本企业关口削减负荷 "
                       f"{d.allocated_mw:.1f} MW") if d.active
                      else f"该行动要求已{_status_cn(d.status)}，无需执行",
            "history": [h.to_dict() for h in d.history],
        } for d in self.directives if d.enterprise_id == enterprise_id]

    def active_directives(self, region: str | None = None,
                          window: str | None = None) -> list[Directive]:
        return [d for d in self.directives if d.active
                and (region is None or d.region == region)
                and (window is None or d.window == window)]

    def reconcile(self, active_alert_windows: set[tuple[str, str]],
                  at: datetime,
                  reason: str = "预警解除，调度撤回指令") -> list[Directive]:
        """预警新版本不再覆盖某地区/时窗时，撤回对应有效指令，容量恢复。"""
        out = []
        for d in list(self.directives):
            if d.active and (d.region, d.window) not in active_alert_windows:
                self._transition(d, "withdrawn", "dispatcher", at, reason)
                out.append(d)
        return out

    # ---- 结算 -----------------------------------------------------------

    def settle_window(self, wk: str | datetime, view: SeriesView,
                      at: datetime) -> list[WindowSettlement]:
        """窗口结束后对照企业关口计量结算实际削峰与未履约容量。

        同一企业同一时窗可能有多条指令（退出后补位、缺口追加），
        按企业聚合成一条结算结果，避免把同一次削峰重复计入。
        """
        window = timewin.key(wk if isinstance(wk, datetime)
                             else datetime.fromisoformat(wk))
        groups: dict[str, list[Directive]] = {}
        for d in self.directives:
            if d.window == window and d.status != "settled":
                groups.setdefault(d.enterprise_id, []).append(d)
        out: list[WindowSettlement] = []
        for eid, ds in sorted(groups.items()):
            res = self.resources[eid]
            baseline = res.baseline_at(window)
            obs = view.at("enterprise_load",
                          datetime.fromisoformat(window), sector=eid)
            load = obs.value if obs else baseline
            achieved = max(round(baseline - load, 3), 0.0)
            total_alloc = round(sum(x.allocated_mw for x in ds), 3)
            active_alloc = round(sum(x.allocated_mw for x in ds if x.active), 3)
            # 企业退出 / 调度撤回的份额不按未履约考核
            unperformed = round(max(active_alloc - achieved, 0.0), 3)
            for x in ds:
                x.status = "settled"
                x.history.append(StatusEvent(
                    at, "settled", "system",
                    f"窗口结算：实际削峰 {achieved} MW"))
            out.append(WindowSettlement(
                enterprise_id=eid, window=window,
                directive_id="+".join(x.directive_id for x in ds),
                allocated_mw=total_alloc,
                baseline_mw=baseline, actual_mw=load,
                actual_reduction_mw=achieved, unperformed_mw=unperformed))
        return out

    # ---- 持久化 ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "resources": [r.to_dict() for r in self.resources.values()],
            "directives": [d.to_dict() for d in self.directives],
            "locked_out": [
                {"enterprise_id": eid, "window": wk, "reason": why}
                for (eid, wk), why in self._locked_out.items()
            ],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ResponseCoordinator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        c = cls()
        for rd in payload["resources"]:
            c.register(Resource.from_dict(rd))
        c.directives = [Directive.from_dict(d) for d in payload["directives"]]
        for item in payload.get("locked_out", []):
            c._locked_out[(item["enterprise_id"], item["window"])] = item["reason"]
        return c


def _status_cn(status: str) -> str:
    return {"opt_out": "由企业退出", "withdrawn": "被调度撤回",
            "settled": "结算完成"}.get(status, status)
