"""Regression detection: baseline run vs candidate run."""
from __future__ import annotations
from .runner import per_scenario


def compare(base, cand, drop=0.34, cost_pct=0.25, latency_pct=0.50) -> dict:
    b, c = per_scenario(base), per_scenario(cand)
    quality, improved, eff = [], [], []
    for sid in sorted(set(b) & set(c)):
        d = b[sid]["pass_rate"] - c[sid]["pass_rate"]
        if d >= drop - 1e-9:
            quality.append({"scenario": sid, "base": b[sid]["pass_rate"], "cand": c[sid]["pass_rate"],
                            "failed_checks": c[sid]["failed_checks"]})
        elif d <= -drop + 1e-9:
            improved.append(sid)
        for k, thr, name in (("cost", cost_pct, "cost"), ("latency", latency_pct, "latency")):
            x, y = b[sid][k], c[sid][k]
            if x and y and (y - x) / x > thr:
                eff.append({"scenario": sid, "metric": name, "base": x, "cand": y, "pct": (y - x) / x})
    return {"quality_regressions": quality, "improvements": improved, "efficiency_regressions": eff,
            "flagged": bool(quality)}
