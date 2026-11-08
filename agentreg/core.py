"""Scenarios, traces, tool environment, scoring. Stdlib only."""
from __future__ import annotations
import json, re, statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Scenario:
    id: str
    user: str
    tools: dict            # name -> {description, input_schema, responses:[{match,result}], default}
    checks: list
    system: str = ""
    tags: list = field(default_factory=list)

    @staticmethod
    def load_dir(d: Path) -> list["Scenario"]:
        out = []
        for p in sorted(Path(d).glob("*.json")):
            j = json.loads(p.read_text())
            out.append(Scenario(j["id"], j["user"], j.get("tools", {}), j["checks"],
                                j.get("system", ""), j.get("tags", [])))
        return out


class ToolEnv:
    """Canned, deterministic tool responses defined by the scenario."""
    def __init__(self, scenario: Scenario):
        self.tools = scenario.tools
        self.calls: list[dict] = []

    def call(self, name: str, args: dict) -> dict:
        self.calls.append({"tool": name, "args": args})
        spec = self.tools.get(name)
        if spec is None:
            return {"error": f"unknown tool {name}"}
        for r in spec.get("responses", []):
            if all(str(args.get(k)) == str(v) for k, v in r["match"].items()):
                return r["result"]
        return spec.get("default", {"error": "no canned response"})


@dataclass
class Trace:
    scenario_id: str
    trial: int
    tool_calls: list        # [{tool,args,result}]
    final: str
    steps: int              # model turns
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str
    simulated: bool = False
    error: str | None = None


# ---------- scoring ----------
def _subset(want: dict, got: dict) -> bool:
    for k, v in want.items():
        if k not in got:
            return False
        if isinstance(v, (int, float)) and isinstance(got[k], (int, float)):
            if abs(float(v) - float(got[k])) > 1e-6:
                return False
        elif str(got[k]).strip().lower() != str(v).strip().lower():
            return False
    return True


def run_check(c: dict, t: Trace) -> bool:
    ty = c["type"]
    names = [x["tool"] for x in t.tool_calls]
    fin = t.final.lower()
    if ty == "tool_called":
        return any(x["tool"] == c["tool"] and _subset(c.get("args", {}), x["args"]) for x in t.tool_calls)
    if ty == "tool_not_called":
        return c["tool"] not in names
    if ty == "tool_before":
        if c["first"] not in names or c["then"] not in names:
            return c["then"] not in names  # vacuous if 'then' never happens
        return names.index(c["first"]) < names.index(c["then"])
    if ty == "final_contains":
        return any(s.lower() in fin for s in c["any"]) if "any" in c else all(s.lower() in fin for s in c["all"])
    if ty == "final_not_contains":
        return not any(s.lower() in fin for s in c["any"])
    if ty == "max_tool_calls":
        return len(names) <= c["n"]
    if ty == "max_steps":
        return t.steps <= c["n"]
    if ty == "no_error":
        return t.error is None
    raise ValueError(f"unknown check type {ty}")


def score(sc: Scenario, t: Trace) -> dict:
    res, wsum, wok, passed = [], 0.0, 0.0, t.error is None
    for c in sc.checks:
        w = float(c.get("weight", 1)); req = c.get("required", True)
        ok = run_check(c, t) if t.error is None else False
        res.append({"check": c["type"], "desc": c.get("desc", ""), "ok": ok, "required": req})
        wsum += w; wok += w if ok else 0
        if req and not ok:
            passed = False
    return {"passed": passed, "score": wok / wsum if wsum else 1.0, "checks": res}


# ---------- cost ----------
def load_pricing() -> dict:
    p = ROOT / "pricing.json"
    return json.loads(p.read_text()) if p.exists() else {}


def cost_usd(t: Trace, pricing: dict) -> float | None:
    pr = pricing.get(t.model)
    if not pr:
        return None
    return (t.input_tokens * pr["in_per_mtok"] + t.output_tokens * pr["out_per_mtok"]) / 1e6


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def pctl(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]
