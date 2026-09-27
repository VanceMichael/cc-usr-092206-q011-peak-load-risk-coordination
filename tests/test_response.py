import unittest
from datetime import datetime, timezone

from src.models import (
    Alert,
    Enterprise,
    InstructionStatus,
    Severity,
    Window,
)
from src.response import ResponseCoordinator

UTC = timezone.utc
NOW = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)
WINDOW = Window(datetime(2026, 7, 14, 14, tzinfo=UTC))


def make_alert(margin=-30.0, region="east", window=WINDOW):
    return Alert(
        id="AL-test",
        region=region,
        window=window,
        severity=Severity.CRITICAL,
        forecast_version=1,
        predicted_load_mw=950.0,
        available_capacity_mw=920.0,
        reserve_margin_mw=margin,
        reserve_margin_ratio=margin / 950.0,
        explanation=(),
        assumptions=(),
        created_at=NOW,
    )


def make_coordinator():
    return ResponseCoordinator(
        (
            Enterprise("E1", "east", 25.0),
            Enterprise("E2", "east", 20.0),
            Enterprise("E3", "east", 10.0),
            Enterprise("E4", "west", 50.0),
        )
    )


class ResponseCoordinatorTest(unittest.TestCase):
    def test_plan_allocates_gap_by_capacity(self):
        coord = make_coordinator()
        issued = coord.plan(make_alert(), NOW)
        self.assertEqual(len(issued), 2)
        self.assertEqual(issued[0].enterprise_id, "E1")
        self.assertAlmostEqual(issued[0].requested_reduction_mw, 25.0)
        self.assertEqual(issued[1].enterprise_id, "E2")
        self.assertAlmostEqual(issued[1].requested_reduction_mw, 5.0)
        self.assertAlmostEqual(coord.committed_mw("east", WINDOW), 30.0)
        self.assertAlmostEqual(coord.available_pool_mw("east", WINDOW), 25.0)

    def test_plan_does_not_overallocate_on_repeat(self):
        coord = make_coordinator()
        coord.plan(make_alert(), NOW)
        again = coord.plan(make_alert(), NOW)
        self.assertEqual(again, ())
        self.assertAlmostEqual(coord.committed_mw("east", WINDOW), 30.0)

    def test_plan_ignores_other_regions(self):
        coord = make_coordinator()
        issued = coord.plan(make_alert(margin=-40.0, region="west"), NOW)
        self.assertEqual(len(issued), 1)
        self.assertEqual(issued[0].enterprise_id, "E4")

    def test_enterprise_exit_restores_capacity_promptly(self):
        coord = make_coordinator()
        issued = coord.plan(make_alert(), NOW)
        coord.acknowledge(issued[0].id, NOW)
        before = coord.committed_mw("east", WINDOW)
        coord.enterprise_exit(issued[1].id, NOW, reason="生产计划变更")
        after = coord.committed_mw("east", WINDOW)
        self.assertAlmostEqual(before - after, 5.0)
        self.assertAlmostEqual(coord.available_pool_mw("east", WINDOW), 30.0)
        restored = [e for e in coord.audit_log() if e[1] == "capacity-restored"]
        self.assertEqual(len(restored), 1)
        self.assertAlmostEqual(restored[0][2]["restored_mw"], 5.0)
        self.assertEqual(restored[0][2]["cause"], "enterprise-exit")
        self.assertEqual(coord.get(issued[1].id).status, InstructionStatus.EXITED)

    def test_dispatch_withdraw_restores_capacity(self):
        coord = make_coordinator()
        issued = coord.plan(make_alert(), NOW)
        coord.dispatch_withdraw(issued[0].id, NOW, reason="负荷回落")
        self.assertAlmostEqual(coord.committed_mw("east", WINDOW), 5.0)
        self.assertEqual(coord.get(issued[0].id).status, InstructionStatus.WITHDRAWN)
        restored = [e for e in coord.audit_log() if e[1] == "capacity-restored"]
        self.assertEqual(restored[0][2]["cause"], "dispatch-withdraw")

    def test_terminal_instruction_cannot_change(self):
        coord = make_coordinator()
        issued = coord.plan(make_alert(), NOW)
        coord.enterprise_exit(issued[0].id, NOW)
        with self.assertRaises(ValueError):
            coord.acknowledge(issued[0].id, NOW)
        with self.assertRaises(ValueError):
            coord.dispatch_withdraw(issued[0].id, NOW)

    def test_record_outcome_fulfilled_and_partial(self):
        coord = make_coordinator()
        issued = coord.plan(make_alert(), NOW)
        done = coord.record_outcome(issued[0].id, 25.0, NOW)
        self.assertEqual(done.status, InstructionStatus.FULFILLED)
        partial = coord.record_outcome(issued[1].id, 3.0, NOW)
        self.assertEqual(partial.status, InstructionStatus.PARTIAL)
        self.assertAlmostEqual(partial.actual_shed_mw, 3.0)

    def test_enterprise_view_is_isolated(self):
        coord = make_coordinator()
        coord.plan(make_alert(margin=-60.0), NOW)  # 需要 E1+E2+E3 全部
        e1_view = coord.enterprise_view("E1")
        self.assertTrue(e1_view)
        self.assertTrue(all(i.enterprise_id == "E1" for i in e1_view))
        self.assertTrue(all(i.enterprise_id == "E3" for i in coord.enterprise_view("E3")))
        self.assertEqual(coord.enterprise_view("E4"), ())

    def test_instruction_history_is_auditable(self):
        coord = make_coordinator()
        issued = coord.plan(make_alert(), NOW)
        coord.acknowledge(issued[0].id, NOW)
        events = [event for _, event in coord.get(issued[0].id).history]
        self.assertEqual(events, ["issued", "acknowledged"])


if __name__ == "__main__":
    unittest.main()
