"""平台核心规则测试。"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src import timewin
from src.ledger import Ledger, ObsFlag
from src.forecast import (Forecaster, ForecastAssumptions, ForecastVersion)
from src.alerts import AlertEngine, ReservePolicy
from src.response import ResponseCoordinator, Resource
from src.review import Reviewer

D = "2026-08-17"
def t(h, m=0):
    return datetime(2026, 8, 17, h, m)
WS = [t(13, 45), t(14, 0), t(14, 15)]
WK = [timewin.key(w) for w in WS]
CAP = {"donghai": 5200.0}
PLAN = {WK[0]: 4720.0, WK[1]: 4735.0, WK[2]: 4720.0}


def seed_minimal(led: Ledger):
    for i, w in enumerate(WS):
        led.record(f"plan-{i}", "planned_load", w, PLAN[WK[i]], "donghai",
                   received_at=t(8))
        for sec, val in (("data_center", 900.0), ("ev_charge", 200.0),
                         ("high_end_manufacturing", 1100.0),
                         ("residential", 1600.0)):
            led.record(f"ind-{sec}-{i}", "industry_load", w, val, "donghai",
                       received_at=t(8, 5), sector=sec)
        led.record(f"evb-{i}", "ev_demand", w, 40.0, "donghai",
                   received_at=t(8, 10))
        led.record(f"mt-{i}", "maintenance", w, 120.0, "donghai",
                   received_at=t(7, 30))


class LedgerTest(unittest.TestCase):
    def test_correction_appends_never_overwrites(self):
        led = Ledger()
        led.record("o1", "regional_load", WS[0], 100.0, "donghai",
                   received_at=t(9))
        led.correct("o2", "o1", 90.0, received_at=t(10), note="倍率更正")
        self.assertEqual(led.get("o1").value, 100.0)   # 原值保留
        self.assertEqual(led.get("o2").supersedes, "o1")
        v = led.view()
        self.assertEqual(v.at("regional_load", WS[0]).obs_id, "o2")
        # 以更正前时刻回看，仍是原读数
        self.assertEqual(
            v.at("regional_load", WS[0], as_of=t(9, 30)).obs_id, "o1")

    def test_late_reading_and_flag_as_of(self):
        led = Ledger()
        led.record("o1", "regional_load", WS[1], 100.0, "donghai",
                   received_at=t(13))
        led.record("late", "regional_load", WS[1], 500.0, "donghai",
                   received_at=t(14, 2), note="迟到")
        v = led.view(as_of=t(14, 0))
        self.assertEqual(v.at("regional_load", WS[1]).value, 100.0)
        v2 = led.view(as_of=t(14, 5))
        self.assertEqual(v2.at("regional_load", WS[1]).value, 500.0)
        led.flag("late", ObsFlag.ANOMALOUS, marked_at=t(14, 20),
                 reason="异常")
        # 标记之前的版本仍可复现该读数；之后被排除
        self.assertIsNotNone(led.view(as_of=t(14, 10)).at(
            "regional_load", WS[1]))
        self.assertEqual(led.view(as_of=t(14, 25)).at(
            "regional_load", WS[1]).obs_id, "o1")

    def test_daily_extremes_are_derived_only(self):
        led = Ledger()
        led.record("a", "regional_load", t(3), 100.0, "donghai",
                   received_at=t(4))
        led.record("b", "regional_load", t(20), 300.0, "donghai",
                   received_at=t(21))
        ext = led.view().daily_extremes("regional_load", "donghai", t(12))
        self.assertEqual(ext["max"][1].value, 300.0)
        self.assertEqual(ext["min"][1].value, 100.0)
        # 原始观测未被派生计算改动
        self.assertEqual(led.get("a").value, 100.0)

    def test_roundtrip(self):
        led = Ledger()
        seed_minimal(led)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "l.json"
            led.save(p)
            led2 = Ledger.load(p)
            self.assertEqual(len(led2.all_observations()),
                             len(led.all_observations()))


class ForecastTest(unittest.TestCase):
    def setUp(self):
        self.led = Ledger()
        seed_minimal(self.led)
        self.fc = Forecaster(ForecastAssumptions())

    def test_incremental_recompute_only_affected(self):
        v1 = self.fc.run("V1", self.led.view(), WS, ["donghai"], t(13, 30),
                         trigger="scheduled", static_baseline=PLAN)
        self.led.record("late", "regional_load", WS[1], 4990.0, "donghai",
                        received_at=t(14, 2))
        affected = self.fc.affected_windows_for(["late"], self.led.view(), WS)
        self.assertEqual(affected, [WK[1], WK[2]])  # 持续项影响两个时窗
        v2 = self.fc.run("V2", self.led.view(), WS, ["donghai"], t(14, 3),
                         trigger="late_reading", affected_windows=affected,
                         parent=v1, static_baseline=PLAN)
        self.assertFalse(v2.get("donghai", WS[0]).recomputed)
        self.assertEqual(v2.get("donghai", WS[0]).carried_from, "V1")
        self.assertTrue(v2.get("donghai", WS[1]).recomputed)
        # W0 数值完全沿用
        self.assertEqual(v1.get("donghai", WS[0]).forecast_mw,
                         v2.get("donghai", WS[0]).forecast_mw)
        # W1 因迟到高读数而抬升
        self.assertGreater(v2.get("donghai", WS[1]).forecast_mw,
                           v1.get("donghai", WS[1]).forecast_mw)

    def test_version_immutable_and_as_of(self):
        v1 = self.fc.run("V1", self.led.view(), WS, ["donghai"], t(13, 30),
                         static_baseline=PLAN)
        w1_before = v1.get("donghai", WS[1]).forecast_mw
        # 14:02 才到的读数
        self.led.record("late", "regional_load", WS[1], 4990.0, "donghai",
                        received_at=t(14, 2))
        v2 = self.fc.run("V2", self.led.view(), WS, ["donghai"], t(14, 3),
                         parent=v1, static_baseline=PLAN)
        # 旧版本冻结不变
        self.assertEqual(v1.get("donghai", WS[1]).forecast_mw, w1_before)
        self.assertGreater(v2.get("donghai", WS[1]).forecast_mw, w1_before)

    def test_typhoon_drop_flagged_transient_not_trend(self):
        self.led.record("wx", "weather", WS[1], 0.0, "donghai",
                        received_at=t(13, 50), unit="-",
                        metrics={"temp_c": 27.0, "rainfall_mm": 35.0,
                                 "wind_km_h": 78.0})
        v = self.fc.run("V1", self.led.view(), WS, ["donghai"], t(13, 55),
                        static_baseline=PLAN)
        wf = v.get("donghai", WS[1])
        self.assertIn("typhoon_residential_drop", wf.contributions_mw)
        self.assertLess(wf.contributions_mw["typhoon_residential_drop"], 0)
        self.assertIn("typhoon_residential_drop", wf.transient_flags)
        # 长周期增长项独立存在
        self.assertGreater(wf.contributions_mw["data_center_growth"], 0)
        # W2（14:15）在天气到达前生成版本时不受影响：用 13:55 的 as_of，
        # wx 影响范围只到其后的两个时窗……这里验证 as_of 截止
        self.assertNotIn("typhoon_residential_drop",
                         v.get("donghai", WS[0]).contributions_mw)

    def test_version_roundtrip(self):
        v1 = self.fc.run("V1", self.led.view(), WS, ["donghai"], t(13, 30),
                         static_baseline=PLAN)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.json"
            v1.save(p)
            back = ForecastVersion.load(p)
            self.assertEqual(back.version_id, "V1")
            self.assertEqual(back.get("donghai", WS[1]).forecast_mw,
                             v1.get("donghai", WS[1]).forecast_mw)


class AlertTest(unittest.TestCase):
    def setUp(self):
        self.led = Ledger()
        seed_minimal(self.led)
        self.fc = Forecaster(ForecastAssumptions())
        self.engine = AlertEngine(CAP, ReservePolicy())

    def test_alert_levels_and_traceability(self):
        v = self.fc.run("V1", self.led.view(), WS, ["donghai"], t(13, 30),
                        static_baseline=PLAN)
        b = self.engine.evaluate("AB1", v, self.led.view(), WS, t(13, 31))
        self.assertTrue(b.alerts)
        a = b.alerts[0]
        # 解释项可追溯到具体 obs_id 与假设
        explained = a.explain()
        self.assertTrue(any(it["item"] == "长周期增长压力" for it in explained))
        self.assertTrue(a.source_obs["baseline"])
        self.assertIn("reserve_ratios", a.assumptions)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.json"
            b.save(p)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(data["batch_id"], "AB1")

    def test_red_when_capacity_tight(self):
        eng = AlertEngine({"donghai": 4900.0}, ReservePolicy())
        v = self.fc.run("V1", self.led.view(), WS, ["donghai"], t(13, 30),
                        static_baseline=PLAN)
        b = eng.evaluate("AB1", v, self.led.view(), WS, t(13, 31))
        self.assertIn(b.alerts[0].level, {"red", "orange"})


class ResponseTest(unittest.TestCase):
    def _coord(self):
        c = ResponseCoordinator()
        c.register(Resource("E1", "算力中心", "donghai",
                            offers={WK[1]: 100.0}, baseline={WK[1]: 600.0}))
        c.register(Resource("E2", "制造厂", "donghai",
                            offers={WK[1]: 100.0}, baseline={WK[1]: 400.0}))
        return c

    def _alert(self, gap):
        from src.alerts import Alert
        return Alert(
            alert_id="AL1", region="donghai", window=WK[1], level="red",
            forecast_mw=5000.0, supply_mw=4900.0, reserve_mw=-100.0,
            reserve_ratio=-0.02, gap_mw=gap, growth_pressure_mw=100.0,
            transient_mw=0.0, transient_flags=[], forecast_version="V1",
            source_obs={}, assumptions={}, explanation=[],
            maintenance_derate_mw=0.0, issued_at=t(13, 35))

    def test_allocate_proportional_and_isolated_view(self):
        c = self._coord()
        ds = c.allocate_for_alerts([self._alert(80.0)], t(13, 35))
        # 同容量企业按 1:1 分摊
        got = {d.enterprise_id: d.allocated_mw for d in ds}
        self.assertAlmostEqual(got["E1"] + got["E2"], 80.0, places=1)
        self.assertAlmostEqual(got["E1"], got["E2"], places=1)
        # 视图隔离
        e1 = c.enterprise_view("E1")
        self.assertEqual({x["window"] for x in e1}, {WK[1]})
        self.assertFalse(any("E2" in json.dumps(x, ensure_ascii=False)
                             for x in e1))

    def test_opt_out_restores_capacity_and_redistributes(self):
        c = self._coord()
        c.allocate_for_alerts([self._alert(120.0)], t(13, 35))
        used_before = sum(d.allocated_mw for d in c.active_directives())
        self.assertAlmostEqual(used_before, 120.0, places=1)
        # E2 退出后容量恢复，缺口由 E1 在其 offer 上限内补足
        c.enterprise_opt_out("E2", t(14, 0))
        refill = c.allocate_for_alerts([self._alert(120.0)], t(14, 1))
        # E1 原有 60，容量上限 100，最多再补 40
        e1_new = sum(d.allocated_mw for d in refill
                     if d.enterprise_id == "E1")
        self.assertLessEqual(
            sum(d.allocated_mw for d in c.active_directives()
                if d.enterprise_id == "E1"), 100.0 + 1e-6)
        self.assertGreater(e1_new, 0)
        # E2 被锁定，不再获得新分配
        self.assertFalse(any(d.enterprise_id == "E2" for d in refill))
        # E2 视图显示已退出、无需执行
        self.assertTrue(all(x["status"] == "opt_out"
                            for x in c.enterprise_view("E2")))

    def test_withdraw_frees_capacity(self):
        c = self._coord()
        ds = c.allocate_for_alerts([self._alert(60.0)], t(13, 35))
        did = ds[0].directive_id
        c.withdraw(did, t(13, 40), reason="测试撤回")
        self.assertFalse(c._find(did).active)
        self.assertEqual(c._find(did).status, "withdrawn")

    def test_no_double_allocation_after_reconcile(self):
        c = self._coord()
        c.allocate_for_alerts([self._alert(60.0)], t(13, 35))
        # 预警解除：全部撤回
        c.reconcile(set(), t(13, 50))
        self.assertEqual(c.active_directives(), [])
        # 再次对账同一空预警集不会产生变化
        again = c.reconcile(set(), t(13, 55))
        self.assertEqual(again, [])

    def test_settlement_unperformed(self):
        c = self._coord()
        c.allocate_for_alerts([self._alert(100.0)], t(13, 35))
        led = Ledger()
        # E1 削峰 50（600→550），E2 完全未履约
        led.record("m1", "enterprise_load", WS[1], 550.0, "donghai",
                   received_at=t(14, 30), sector="E1")
        led.record("m2", "enterprise_load", WS[1], 400.0, "donghai",
                   received_at=t(14, 30), sector="E2")
        res = c.settle_window(WS[1], led.view(), t(14, 35))
        by = {s.enterprise_id: s for s in res}
        self.assertAlmostEqual(by["E1"].actual_reduction_mw, 50.0, places=1)
        self.assertAlmostEqual(by["E1"].unperformed_mw, 0.0, places=1)
        self.assertAlmostEqual(by["E2"].actual_reduction_mw, 0.0, places=1)
        self.assertAlmostEqual(by["E2"].unperformed_mw, 50.0, places=1)

    def test_dispatch_withdraw_can_be_recalled_later(self):
        # 调度撤回（非企业退出）：预警重现时企业资源可再次被调用
        c = self._coord()
        c.allocate_for_alerts([self._alert(60.0)], t(13, 35))
        c.reconcile(set(), t(13, 50))  # 预警消失，全部调度撤回
        again = c.allocate_for_alerts([self._alert(60.0)], t(14, 0))
        self.assertTrue(any(d.enterprise_id == "E2" for d in again))

    def test_roundtrip(self):
        c = self._coord()
        c.allocate_for_alerts([self._alert(40.0)], t(13, 35))
        c.enterprise_opt_out("E2", t(14, 0))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            c.save(p)
            c2 = ResponseCoordinator.load(p)
            self.assertEqual(len(c2.directives), len(c.directives))
            self.assertEqual(c2.available("E1", WK[1]), c.available("E1", WK[1]))


class ReviewTest(unittest.TestCase):
    def test_false_alert_from_anomalous_obs(self):
        led = Ledger()
        seed_minimal(led)
        fc = Forecaster(ForecastAssumptions())
        engine = AlertEngine(CAP, ReservePolicy())
        # 偏高迟到读数触发红警
        led.record("bad", "regional_load", WS[1], 5300.0, "donghai",
                   received_at=t(14, 2))
        v = fc.run("V1", led.view(), WS, ["donghai"], t(14, 3),
                   affected_windows=[WK[1]], static_baseline=PLAN)
        b = engine.evaluate("AB1", v, led.view(), WS, t(14, 4))
        self.assertTrue(any(a.level == "red" for a in b.alerts))
        # 事后标记异常 + 实际负荷证明备用充裕
        led.flag("bad", ObsFlag.ANOMALOUS, marked_at=t(15),
                 reason="倍率错误")
        led.record("actual", "regional_load", WS[1], 4600.0, "donghai",
                   received_at=t(15, 10))
        coord = ResponseCoordinator()
        rev = Reviewer(led, coord, CAP).run("R1", b, v, [WS[1]], t(15, 30))
        self.assertTrue(rev.false_alerts)
        self.assertIn("bad", rev.false_alerts[0].offending_obs)
        data = rev.to_dict()
        self.assertIn("totals", data)


class PlatformFlowTest(unittest.TestCase):
    def test_refresh_incremental_and_reconcile_withdraws(self):
        from src.platform import Platform
        p = Platform(CAP, ["donghai"], WS)
        seed_minimal(p.ledger)
        p.coordinator.register(Resource("E1", "算力中心", "donghai",
                                        offers={wk: 2000.0 for wk in WK},
                                        baseline={wk: 600.0 for wk in WK}))
        # 例行预测：红警并分配
        fv1, b1 = p.refresh(t(13, 30), trigger="scheduled",
                            static_baseline=PLAN)
        self.assertTrue(p.coordinator.active_directives())
        covered_before = {d.window for d in p.coordinator.active_directives()}
        # 迟到读数：只重算两个时窗
        p.ingest("late", "regional_load", WS[1], 4990.0, "donghai",
                 received_at=t(14, 2))
        fv2, b2 = p.refresh(t(14, 3), trigger="late_reading",
                            changed_obs=["late"], static_baseline=PLAN)
        self.assertEqual(fv2.get("donghai", WS[0]).carried_from, fv1.version_id)
        # 更正后备用全面充裕：新一轮预警为空，既有有效指令被对账撤回
        p.correct("fix", "late", 3000.0, t(14, 20), reason="倍率错误")
        p.ledger.flag("late", ObsFlag.ANOMALOUS, marked_at=t(14, 21))
        fv3, b3 = p.refresh(t(14, 22), trigger="correction",
                            changed_obs=["fix", "late"], static_baseline=PLAN)
        # W1/W2 预警消失 → 指令撤回；W0 时窗不在更正影响范围，红警保留
        active_windows = {d.window for d in p.coordinator.active_directives()}
        self.assertEqual(active_windows, {WK[0]})
        withdrawn = [d for d in p.coordinator.directives
                     if d.status == "withdrawn" and d.window in (WK[1], WK[2])]
        self.assertTrue(withdrawn)
        self.assertEqual(len(p.versions), 3)
        # 旧版本不可变
        self.assertIsNotNone(fv1.get("donghai", WS[1]))


if __name__ == "__main__":
    unittest.main()
