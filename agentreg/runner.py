from __future__ import annotations
import json, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from .core import ROOT, Scenario, Trace, score, cost_usd, load_pricing, med, pctl


def run_suite(agent, scenarios, trials=1, workers=4, label="run") -> dict:
    pricing = load_pricing()
    jobs = [(sc, k) for sc in scenarios for k in range(trials)]
    def one(j):
        sc, k = j
        t = agent(sc, k); s = score(sc, t)
        return {"trace": asdict(t), "score": s, "cost_usd": cost_usd(t, pricing)}
    wall = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(one, jobs))
    return {"label": label, "trials": trials, "wall_s": round(time.perf_counter() - wall, 3),
            "results": results, "summary": summarize(results)}


def summarize(results) -> dict:
    lat = [r["trace"]["latency_s"] for r in results]
    cost = [r["cost_usd"] for r in results]
    n = len(results)
    return {"runs": n, "pass_rate": sum(r["score"]["passed"] for r in results) / n if n else 0,
            "mean_score": sum(r["score"]["score"] for r in results) / n if n else 0,
            "latency_p50_s": med(lat), "latency_p95_s": pctl(lat, .95),
            "cost_per_run_usd": (sum(c for c in cost if c is not None) / max(1, sum(c is not None for c in cost))
                                 if any(c is not None for c in cost) else None),
            "cost_p95_usd": pctl(cost, .95),
            "tokens_per_run": sum(r["trace"]["input_tokens"] + r["trace"]["output_tokens"] for r in results) / n if n else 0,
            "simulated": all(r["trace"]["simulated"] for r in results) if n else False}


def per_scenario(run) -> dict:
    d = {}
    for r in run["results"]:
        d.setdefault(r["trace"]["scenario_id"], []).append(r)
    out = {}
    for sid, rs in d.items():
        out[sid] = {"pass_rate": sum(x["score"]["passed"] for x in rs) / len(rs),
                    "score": sum(x["score"]["score"] for x in rs) / len(rs),
                    "cost": med([x["cost_usd"] for x in rs]), "latency": med([x["trace"]["latency_s"] for x in rs]),
                    "failed_checks": sorted({c["check"] + (":" + c["desc"] if c["desc"] else "")
                                             for x in rs for c in x["score"]["checks"] if not c["ok"] and c["required"]})}
    return out


def save(run, label=None) -> Path:
    p = ROOT / "runs" / f"{label or run['label']}.json"
    p.write_text(json.dumps(run, indent=1)); return p


def load(label) -> dict:
    p = Path(label) if Path(label).exists() else ROOT / "runs" / f"{label}.json"
    return json.loads(p.read_text())
