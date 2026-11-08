"""Agreement between automated scoring and your manual pass/fail labels."""
from __future__ import annotations
import json
from .core import ROOT


def labels_path(run_label): return ROOT / "labels" / f"{run_label}.json"


def load_labels(run_label) -> dict:
    p = labels_path(run_label)
    return json.loads(p.read_text()) if p.exists() else {}


def agreement(run, labels: dict) -> dict:
    """labels: {"scenario_id#trial": "pass"|"fail"} (trial 0 if no '#')."""
    pairs, dis = [], []
    for r in run["results"]:
        t = r["trace"]; key = f"{t['scenario_id']}#{t['trial']}"
        lab = labels.get(key) or (labels.get(t["scenario_id"]) if t["trial"] == 0 else None)
        if lab not in ("pass", "fail"):
            continue
        auto = "pass" if r["score"]["passed"] else "fail"
        pairs.append((auto, lab))
        if auto != lab:
            dis.append({"key": key, "auto": auto, "manual": lab,
                        "failed_checks": [c["check"] for c in r["score"]["checks"] if not c["ok"]]})
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    tp = sum(a == "pass" and m == "pass" for a, m in pairs); tn = sum(a == "fail" and m == "fail" for a, m in pairs)
    fp = sum(a == "pass" and m == "fail" for a, m in pairs)  # auto too lenient
    fn = sum(a == "fail" and m == "pass" for a, m in pairs)  # auto too strict
    po = (tp + tn) / n
    pa, pm = (tp + fp) / n, (tp + fn) / n
    pe = pa * pm + (1 - pa) * (1 - pm)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": n, "agreement": po, "kappa": kappa, "too_lenient": fp, "too_strict": fn, "disagreements": dis}
