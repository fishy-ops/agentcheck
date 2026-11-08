import unittest
from agentreg.agree import agreement

def run(*passed):
    return {"results": [{"trace": {"scenario_id": f"s{i}", "trial": 0},
            "score": {"passed": p, "checks": [{"check": "x", "ok": p}]}} for i, p in enumerate(passed)]}

class T(unittest.TestCase):
    def test_perfect(self):
        r = agreement(run(True, False, True, False), {"s0": "pass", "s1": "fail", "s2": "pass", "s3": "fail"})
        self.assertEqual((r["agreement"], r["kappa"]), (1.0, 1.0))
    def test_disagreements_and_direction(self):
        r = agreement(run(True, True, False, False), {"s0": "pass", "s1": "fail", "s2": "pass", "s3": "fail"})
        self.assertEqual((r["agreement"], r["too_lenient"], r["too_strict"]), (0.5, 1, 1))
        self.assertAlmostEqual(r["kappa"], 0.0)
    def test_unlabelled_skipped(self):
        self.assertEqual(agreement(run(True), {})["n"], 0)

if __name__ == "__main__":
    unittest.main()
