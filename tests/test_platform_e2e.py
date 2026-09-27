"""端到端：最大负荷刷新纪录的值班日。

流程：数据接入 → 初始预测(v1) → 日内预警 → 需求响应（退出/撤回容量恢复）
→ 迟到读数局部重算(v2) → 计量更正局部重算(v3) → 新预测请求(v4)
→ 企业视图隔离 → 预警逐项解释 → 实际值到达 → 事后复盘。
"""

import unittest
from datetime import date, datetime, timezone

from src.forecasts import ForecastConfig
from src.models import (
    INDUSTRY_DATA_CENTER,
    INDUSTRY_HIGH_END_MFG,
    Enterprise,
    InstructionStatus,
    Observation,
    ObservationKind,
    Quality,
    Severity,
    Window,
)
from src.platform import LoadRiskPlatform
from src.risks import CapacityProfile

UTC = timezone.utc
DAY = datetime(2026, 7, 14, tzinfo=UTC)
H12 = Window(DAY.replace(hour=12))
H13 = Window(DAY.replace(hour=13))
H14 = Window(DAY.replace(hour=14))
HORIZON = (H12, H13, H14)
T0 = datetime(2026, 7, 14, 8, 0, tzinfo=UTC)


def meter(obs_id, day, hour, load, ingested_at=None, corrects=None, quality=Quality.NORMAL):
    window = Window(day.replace(hour=hour, minute=0, second=0, microsecond=0))
    return Observation(
        id=obs_id,
        kind=ObservationKind.METER,
        region="east",
        window=window,
        metrics={"load_mw": load},
        observed_at=window.start,
        ingested_at=ingested_at or window.end,
        corrects=corrects,
        quality=quality,
    )


def build_platform():
    platform = LoadRiskPlatform(
        forecast_config=ForecastConfig(
            default_base_mw={"east": 800.0},
            industry_base_mw={
                ("east", INDUSTRY_DATA_CENTER): 100.0,
                ("east", INDUSTRY_HIGH_END_MFG): 150.0,
            },
            industry_growth_daily={
                INDUSTRY_DATA_CENTER: 0.003,
                INDUSTRY_HIGH_END_MFG: 0.001,
            },
            growth_reference_date=date(2026, 7, 1),
        ),
        capacity=(CapacityProfile("east", 1000.0),),
        enterprises=(
            Enterprise("E1", "east", 25.0),
            Enterprise("E2", "east", 20.0),
            Enterprise("E3", "east", 10.0),
        ),
    )
    platform.set_horizon(("east",), HORIZON)
    # 历史计量：数据中心、高端制造增长背景下，基础负荷连日高位
    for d in (11, 12, 13):
        day = datetime(2026, 7, d, tzinfo=UTC)
        for hour in (12, 13, 14):
            platform.ingest(meter(f"hist-{d}-{hour}", day, hour, 820.0))
    # 台风降雨：居民用电短时下降（H12）
    platform.ingest(
        Observation(
            id="wx-rain",
            kind=ObservationKind.WEATHER,
            region="east",
            window=H12,
            metrics={"rain_mm": 25.0, "temp_c": 26.0},
            observed_at=H12.start,
            ingested_at=T0,
        )
    )
    # 充换电需求集中申报（H13），该申报事后被证明异常偏高
    platform.ingest(
        Observation(
            id="ev-1",
            kind=ObservationKind.EV_DEMAND,
            region="east",
            window=H13,
            metrics={"requested_mw": 200.0},
            observed_at=H13.start,
            ingested_at=T0,
            quality=Quality.ABNORMAL,
        )
    )
    # 高温（H14）与设备检修（H14 停运 80 MW）
    platform.ingest(
        Observation(
            id="wx-heat",
            kind=ObservationKind.WEATHER,
            region="east",
            window=H14,
            metrics={"temp_c": 38.0, "rain_mm": 0.0},
            observed_at=H14.start,
            ingested_at=T0,
        )
    )
    platform.ingest(
        Observation(
            id="mnt-1",
            kind=ObservationKind.MAINTENANCE,
            region="east",
            window=H14,
            metrics={"outage_mw": 80.0},
            observed_at=H14.start,
            ingested_at=T0,
        )
    )
    return platform


class DutyDayEndToEndTest(unittest.TestCase):
    def setUp(self):
        self.platform = build_platform()

    def test_full_duty_day_flow(self):
        p = self.platform

        # ---- 初始预测 v1：台风降雨只压 H12，增长与高温推高 H13/H14 ----
        v1 = p.run_forecast(T0)
        self.assertEqual(v1.version, 1)
        self.assertAlmostEqual(v1.result_for("east", H12).components["weather"], -49.2)
        self.assertAlmostEqual(v1.result_for("east", H13).predicted_load_mw, 1025.93, places=2)
        self.assertAlmostEqual(v1.result_for("east", H14).predicted_load_mw, 950.93, places=2)

        # ---- 日内预警：H13/H14 触发，H12 不触发 ----
        alerts = p.evaluate_alerts(T0)
        by_window = {a.window: a for a in alerts}
        self.assertEqual(set(by_window), {H13, H14})
        self.assertEqual(by_window[H14].severity, Severity.CRITICAL)

        # ---- 需求响应：按缺口分配；企业退出、调度撤回后容量及时恢复 ----
        i14 = p.plan_response(by_window[H14].id, T0)  # 缺口 30.93 -> E1 25 + E2 5.93
        self.assertEqual([i.enterprise_id for i in i14], ["E1", "E2"])
        self.assertAlmostEqual(p.responses.committed_mw("east", H14), 30.93, places=2)
        p.responses.enterprise_exit(i14[1].id, T0, reason="产线无法停机")
        self.assertAlmostEqual(p.responses.committed_mw("east", H14), 25.0)

        i13 = p.plan_response(by_window[H13].id, T0)  # 缺口 25.93 -> E1 25 + E2 0.93
        p.responses.dispatch_withdraw(i13[1].id, T0, reason="指令重复")
        self.assertAlmostEqual(p.responses.committed_mw("east", H13), 25.0)
        restored = [e for e in p.responses.audit_log() if e[1] == "capacity-restored"]
        self.assertEqual(len(restored), 2)

        # ---- 迟到读数：只重算受影响的 H14 ----
        late = meter(
            "late-1", datetime(2026, 7, 10, tzinfo=UTC), 14, 840.0,
            ingested_at=datetime(2026, 7, 14, 10, 0, tzinfo=UTC),
        )
        result = p.ingest(late)
        self.assertTrue(result.triggers_forecast_recompute)
        self.assertEqual(set(result.affected_keys), {("east", H14)})
        v2 = p.recompute_pending(datetime(2026, 7, 14, 10, 5, tzinfo=UTC))
        self.assertEqual(v2.reason, "late-reading")
        self.assertEqual(set(v2.recomputed_keys), {("east", H14.key)})
        self.assertIs(v2.result_for("east", H13), v1.result_for("east", H13))
        self.assertAlmostEqual(v2.result_for("east", H14).components["base"], 825.0)

        # ---- 计量更正：原始读数不改写，仍只重算 H14 ----
        p.ingest(meter("corr-1", datetime(2026, 7, 10, tzinfo=UTC), 14, 800.0, corrects="late-1"))
        v3 = p.recompute_pending(datetime(2026, 7, 14, 10, 30, tzinfo=UTC))
        self.assertEqual(v3.reason, "correction")
        self.assertEqual(set(v3.recomputed_keys), {("east", H14.key)})
        self.assertAlmostEqual(v3.result_for("east", H14).components["base"], 815.0)
        self.assertEqual(p.store.get("late-1").metrics["load_mw"], 840.0)  # 原始观测仍在
        self.assertEqual(p.store.effective("late-1").metrics["load_mw"], 800.0)

        # ---- 新预测请求：只重算指定窗口 ----
        v4 = p.request_new_forecast(frozenset({("east", H12)}), datetime(2026, 7, 14, 11, 0, tzinfo=UTC))
        self.assertEqual(v4.reason, "new-forecast")
        self.assertEqual(set(v4.recomputed_keys), {("east", H12.key)})
        self.assertEqual(len(p.forecasts.versions), 4)

        # ---- 预警随版本刷新，旧预警留档 ----
        fresh = p.evaluate_alerts(datetime(2026, 7, 14, 11, 5, tzinfo=UTC))
        self.assertEqual({a.window for a in fresh}, {H13, H14})
        self.assertTrue(all(a.forecast_version == 4 for a in p.active_alerts))

        # ---- 企业视图隔离 ----
        self.assertEqual(len(p.enterprise_view("E1")), 2)
        self.assertTrue(all(i.enterprise_id == "E1" for i in p.enterprise_view("E1")))
        self.assertEqual(len(p.enterprise_view("E2")), 2)
        self.assertEqual(p.enterprise_view("E3"), ())

        # ---- 预警逐项解释 ----
        active_h14 = next(a for a in p.active_alerts if a.window == H14)
        doc = p.explain_alert(active_h14.id)
        labels = [item["label"] for item in doc["items"]]
        self.assertIn("分量·数据中心", labels)
        self.assertIn("分量·高端制造", labels)
        capacity_item = next(i for i in doc["items"] if i["label"] == "可用容量")
        self.assertIn("mnt-1", capacity_item["sources"])
        self.assertTrue(any("检修计划" in a for a in doc["assumptions"]))
        self.assertTrue(any("降温负荷" in a for a in doc["assumptions"]))

        # ---- 实际值到达（当日读数不触发重算） ----
        for obs_id, window, load in (
            ("act-12", H12, 770.0),
            ("act-13", H13, 850.0),
            ("act-14", H14, 960.0),
        ):
            hour = window.start.hour
            ingested = p.ingest(meter(obs_id, DAY, hour, load))
            self.assertFalse(ingested.triggers_forecast_recompute)

        # ---- 履约登记：H13 E1 足额，H14 E1 部分 ----
        e1_h13 = next(i for i in p.enterprise_view("E1") if i.window == H13)
        e1_h14 = next(i for i in p.enterprise_view("E1") if i.window == H14)
        p.responses.record_outcome(e1_h13.id, 25.0, datetime(2026, 7, 14, 15, 0, tzinfo=UTC))
        p.responses.record_outcome(e1_h14.id, 20.0, datetime(2026, 7, 14, 16, 0, tzinfo=UTC))
        self.assertEqual(p.responses.get(e1_h13.id).status, InstructionStatus.FULFILLED)
        self.assertEqual(p.responses.get(e1_h14.id).status, InstructionStatus.PARTIAL)

        # ---- 事后复盘 ----
        report = p.review(datetime(2026, 7, 15, 9, 0, tzinfo=UTC))
        self.assertEqual(report.alert_count, 2)
        self.assertEqual(report.false_alarm_count, 1)  # H13：实际 850，裕度充足
        self.assertAlmostEqual(report.total_committed_mw, 25.0 + 30.93, places=2)
        self.assertAlmostEqual(report.total_actual_shed_mw, 45.0)
        self.assertAlmostEqual(report.total_unfulfilled_mw, 10.93, places=2)

        rows = {r.window: r for r in report.rows}
        self.assertTrue(rows[H13].false_alarm)
        causes = "；".join(rows[H13].false_alarm_causes)
        self.assertIn("ev-1", causes)  # 异常充换电申报引发误报
        self.assertFalse(rows[H14].false_alarm)
        self.assertAlmostEqual(rows[H14].unfulfilled_mw, 10.93, places=2)

        # ---- 跨日峰谷只读，原始观测不改写 ----
        peak, valley = p.store.daily_peak_valley("east", date(2026, 7, 14))
        self.assertEqual(peak[1], 960.0)
        self.assertEqual(valley[1], 770.0)
        self.assertEqual(p.store.get("act-14").metrics["load_mw"], 960.0)


if __name__ == "__main__":
    unittest.main()
