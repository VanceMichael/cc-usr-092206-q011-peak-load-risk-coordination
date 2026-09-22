"""台风降雨日全流程演示（合成数据，无真实企业信息）。

场景：最大负荷刷新纪录的值班日。
* 东海区数据中心/充换电/高端制造持续增长，局部备用逼近红线；
* 午后台风降雨造成居民用电短时下降——必须识别为瞬时项而非趋势；
* 一条迟到且偏高的地区计量触发红色误报，事后更正并标记异常；
* 企业 E2 退出响应，其已分配容量即时恢复并改派 E1/E3；
* 调度对 W2 的一条指令主动撤回；
* 窗口结束后复盘实际削峰、未履约与异常数据引发的误报。

运行：python -m examples.typhoon_day
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src import timewin
from src.ledger import Ledger, ObsFlag
from src.forecast import Forecaster, ForecastAssumptions
from src.alerts import AlertEngine, ReservePolicy
from src.response import ResponseCoordinator, Resource
from src.review import Reviewer

DAY = "2026-08-17"
W0 = datetime.fromisoformat(f"{DAY}T13:45")
W1 = datetime.fromisoformat(f"{DAY}T14:00")
W2 = datetime.fromisoformat(f"{DAY}T14:15")
WINDOWS = [W0, W1, W2]

CAPACITY = {"donghai": 5200.0}
# 规划基准典型日曲线
PLAN = {timewin.key(W0): 4720.0, timewin.key(W1): 4735.0,
        timewin.key(W2): 4720.0}


def build_ledger() -> Ledger:
    led = Ledger()
    # 规划基准典型日曲线（当日 08 点前随基准曲线下发；与实测分开）
    for i, w in enumerate(WINDOWS):
        led.record(f"plan-{i}", "planned_load", w, PLAN[timewin.key(w)],
                   "donghai", received_at=f"{DAY}T08:00",
                   note="规划基准典型日曲线")
    # 行业计量：增长行业体量
    sector_levels = {"data_center": 900.0, "ev_charge": 200.0,
                     "high_end_manufacturing": 1100.0,
                     "residential": 1600.0}
    for i, w in enumerate(WINDOWS):
        for sec, val in sector_levels.items():
            led.record(f"ind-{sec}-{i}", "industry_load", w, val,
                       "donghai", received_at=f"{DAY}T08:05", sector=sec)
    # 充换电预约需求（午后补能小高峰）
    for i, w in enumerate(WINDOWS):
        led.record(f"evbook-{i}", "ev_demand", w, 40.0 + i * 5,
                   "donghai", received_at=f"{DAY}T08:10")
    # 检修：一台主变检修扣减 120 MW
    for i, w in enumerate(WINDOWS):
        led.record(f"maint-{i}", "maintenance", w, 120.0, "donghai",
                   received_at=f"{DAY}T07:30", note="2号主变检修")
    # 气象：14:00 起台风过境，暴雨+大风
    led.record("wx-0", "weather", W0, 0.0, "donghai",
               received_at=f"{DAY}T13:00", unit="-",
               metrics={"temp_c": 33.0, "rainfall_mm": 2.0,
                        "wind_km_h": 25.0})
    led.record("wx-1", "weather", W1, 0.0, "donghai",
               received_at=f"{DAY}T14:05", unit="-",
               metrics={"temp_c": 27.0, "rainfall_mm": 35.0,
                        "wind_km_h": 78.0})
    led.record("wx-2", "weather", W2, 0.0, "donghai",
               received_at=f"{DAY}T14:20", unit="-",
               metrics={"temp_c": 27.5, "rainfall_mm": 28.0,
                        "wind_km_h": 70.0})
    return led


def build_coordinator() -> ResponseCoordinator:
    c = ResponseCoordinator()
    c.register(Resource("E1", "东海智算中心", "donghai",
                        offers={timewin.key(w): 220.0 for w in WINDOWS},
                        baseline={timewin.key(w): 600.0 for w in WINDOWS}))
    c.register(Resource("E2", "东海精密制造", "donghai",
                        offers={timewin.key(w): 60.0 for w in WINDOWS},
                        baseline={timewin.key(w): 400.0 for w in WINDOWS}))
    c.register(Resource("E3", "东海充换电聚合商", "donghai",
                        offers={timewin.key(w): 220.0 for w in WINDOWS},
                        baseline={timewin.key(w): 500.0 for w in WINDOWS}))
    return c


def show_alerts(title, batch):
    print(title)
    for a in batch.alerts:
        print(f"  [{a.level:>6}] {a.window[11:16]} 预测 {a.forecast_mw:7.1f} "
              f"/ 供电 {a.supply_mw:7.1f} / 备用率 {a.reserve_ratio:6.2%} "
              f"缺口 {a.gap_mw:6.1f} MW")


def main():
    led = build_ledger()
    coord = build_coordinator()
    forecaster = Forecaster(ForecastAssumptions())

    # ---- 版本 v1：13:30 例行预测 ----------------------------------------
    v1 = forecaster.run("FV-20260817-01", led.view(), WINDOWS, ["donghai"],
                        datetime.fromisoformat(f"{DAY}T13:30"),
                        trigger="scheduled", static_baseline=PLAN,
                        note="台风登陆前例行预测")
    engine = AlertEngine(CAPACITY, ReservePolicy())
    b1 = engine.evaluate("AB-01", v1, led.view(), WINDOWS,
                         datetime.fromisoformat(f"{DAY}T13:31"))
    show_alerts("== 13:31 首轮预警（v1） ==", b1)

    ds1 = coord.allocate_for_alerts(b1.alerts,
                                    datetime.fromisoformat(f"{DAY}T13:35"))
    print("\n== 13:35 下发行动要求 ==")
    for d in ds1:
        print(f"  {d.directive_id} -> {d.enterprise_id} @ {d.window[11:16]} "
              f"{d.allocated_mw:.1f} MW")

    print("\n== E1 企业视图：只能看到本企业的行动要求 ==")
    e1_view = coord.enterprise_view("E1")
    assert all(v["window"] and "required_reduction_mw" in v for v in e1_view)
    assert all("E2" not in str(v) and "E3" not in str(v) for v in e1_view)
    for item in e1_view:
        print(f"  [{item['status']}] {item['action']}")

    # ---- 迟到且偏高的地区计量：只重算受影响时窗，产生 v2 ---------------
    led.record("late-load-1", "regional_load", W1, 4990.0, "donghai",
               received_at=f"{DAY}T14:02", note="迟到的关口读数（事后查明异常）")
    affected = forecaster.affected_windows_for(["late-load-1"],
                                               led.view(), WINDOWS)
    v2 = forecaster.run("FV-20260817-02", led.view(), WINDOWS, ["donghai"],
                        datetime.fromisoformat(f"{DAY}T14:03"),
                        trigger="late_reading", affected_windows=affected,
                        parent=v1, static_baseline=PLAN,
                        note="迟到读数只重算 14:00/14:15 两个时窗")
    assert v2.get("donghai", W0).recomputed is False
    assert v2.get("donghai", W0).carried_from == v1.version_id
    assert v2.get("donghai", W1).recomputed is True
    b2 = engine.evaluate("AB-02", v2, led.view(), WINDOWS,
                         datetime.fromisoformat(f"{DAY}T14:04"))
    show_alerts(f"\n== 14:04 迟到读数后增量预警（重算 {affected}） ==", b2)
    ds2 = coord.allocate_for_alerts(b2.alerts,
                                    datetime.fromisoformat(f"{DAY}T14:06"))
    print("  追加分摊：",
          [(d.directive_id, d.enterprise_id, round(d.allocated_mw, 1))
           for d in ds2])

    # ---- 14:10 E2 退出响应：容量即时恢复并改派 E1/E3 -------------------
    opted = coord.enterprise_opt_out(
        "E2", datetime.fromisoformat(f"{DAY}T14:10"),
        reason="产线突发工艺约束，企业退出响应")
    print(f"\n== 14:10 E2 退出 {len(opted)} 条指令，对应容量即时恢复 ==")
    refill = coord.allocate_for_alerts(b2.alerts,
                                       datetime.fromisoformat(f"{DAY}T14:11"))
    for d in refill:
        print(f"  改派 {d.directive_id} -> {d.enterprise_id} "
              f"{d.allocated_mw:.1f} MW")

    # ---- 14:12 调度主动撤回 E3 在 14:15 时窗的一条指令 ------------------
    w2_e3 = [d for d in coord.directives
             if d.enterprise_id == "E3" and d.window == timewin.key(W2)][0]
    coord.withdraw(w2_e3.directive_id,
                   datetime.fromisoformat(f"{DAY}T14:12"),
                   reason="14:15 充电预约取消，无需聚合商削峰")
    print(f"\n== 14:12 调度撤回 {w2_e3.directive_id}，"
          f"E3 在该时窗视图更新： ==")
    for item in coord.enterprise_view("E3"):
        if item["window"] == timewin.key(W2):
            print(f"  [{item['status']}] {item['action']}")

    # ---- 14:20 计量更正：追加更正读数并标记异常，产生 v3 ---------------
    led.correct("fix-load-1", "late-load-1", 4760.0,
                received_at=f"{DAY}T14:20", note="关口表倍率错误，更正")
    led.flag("late-load-1", ObsFlag.ANOMALOUS, marked_at=f"{DAY}T14:20",
             reason="倍率错误导致读数偏高")
    affected2 = forecaster.affected_windows_for(
        ["fix-load-1", "late-load-1"], led.view(), WINDOWS)
    v3 = forecaster.run("FV-20260817-03", led.view(), WINDOWS, ["donghai"],
                        datetime.fromisoformat(f"{DAY}T14:21"),
                        trigger="correction", affected_windows=affected2,
                        parent=v2, static_baseline=PLAN,
                        note="计量更正后再重算受影响时窗")
    b3 = engine.evaluate("AB-03", v3, led.view(), WINDOWS,
                         datetime.fromisoformat(f"{DAY}T14:22"))
    show_alerts("\n== 14:22 更正后预警（14:00 由红降橙） ==", b3)
    # 对账：新版本若不再覆盖某预警时窗，撤回其指令（本场景时窗仍在，
    # 撤回数为 0；reconcile 的效果由单元测试覆盖）
    withdrawn = coord.reconcile({(a.region, a.window) for a in b3.alerts},
                                datetime.fromisoformat(f"{DAY}T14:23"))
    print(f"  对账撤回指令 {len(withdrawn)} 条")

    # 分析人员逐项解释
    a = next(a for a in b3.alerts if a.window == timewin.key(W1))
    print("\n== 14:00 橙色预警逐项解释 ==")
    for line in a.explanation:
        print("  " + line)
    print("  数据溯源（用途 -> obs_id）：")
    for use, ids in a.source_obs.items():
        print(f"    {use:14s}: {ids}")

    print("\n== 14:00 预测贡献分解：长周期趋势与瞬时项分列 ==")
    for k, v in v3.get("donghai", W1).contributions_mw.items():
        tag = "  [瞬时项-台风，不得外推为趋势]" if k.startswith("typhoon_") else ""
        print(f"  {k:28s} {v:+8.1f} MW{tag}")

    # ---- 窗口结束：实际计量进入台账（原始观测，不被任何版本改写） -------
    led.record("act-E1", "enterprise_load", W1, 405.0, "donghai",
               received_at=f"{DAY}T14:35", sector="E1", note="E1 实际负荷")
    led.record("act-E2", "enterprise_load", W1, 400.0, "donghai",
               received_at=f"{DAY}T14:35", sector="E2", note="E2 已退出")
    led.record("act-E3", "enterprise_load", W1, 315.0, "donghai",
               received_at=f"{DAY}T14:35", sector="E3", note="E3 实际负荷")
    led.record("act-region", "regional_load", W1, 4610.0, "donghai",
               received_at=f"{DAY}T15:00",
               note="14:00 实际地区负荷（台风居民下降，事后确认）")

    # ---- 复盘：对照触发误报的 v2 / AB-02 批次 ---------------------------
    reviewer = Reviewer(led, coord, CAPACITY)
    report = reviewer.run("RPT-20260817-1", b2, v2, [W1],
                          datetime.fromisoformat(f"{DAY}T15:30"))
    print("\n== 事后复盘 ==")
    for line in report.summary():
        print("  " + line)
    print("  履约明细（14:00 时窗）：")
    for s in report.settlements:
        print(f"    {s.enterprise_id}: 分配 {s.allocated_mw:6.1f} / "
              f"实际削峰 {s.actual_reduction_mw:6.1f} / "
              f"未履约 {s.unperformed_mw:5.1f} MW")

    # ---- 持久化：版本、台账、指令、复盘均可按版本回溯 -------------------
    v3.save("/tmp/peak-fv03.json")
    led.save("/tmp/peak-ledger.json")
    coord.save("/tmp/peak-coordinator.json")
    b3.save("/tmp/peak-alerts03.json")
    Path("/tmp/peak-review.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("\n已持久化到 /tmp/peak-*.json：月报形成前可回溯任意预测版本"
          "与当时依据的观测。")


if __name__ == "__main__":
    main()
