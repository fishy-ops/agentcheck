"""Seeded-regression benchmark: inject KNOWN defects into the mock agent and measure
how many the suite catches (recall) and how often a benign change is wrongly flagged."""
from __future__ import annotations
from .agents import MockAgent
from .compare import compare
from .runner import run_suite

# (name, defects, should_flag_quality, scenarios that must be implicated)
SEEDED = [
    ("skip_lookup",  {"skip_lookup"},  True,  {"refund_eligible"}),
    ("over_refund",  {"over_refund"},  True,  {"refund_ineligible"}),
    ("no_confirm",   {"no_confirm"},   True,  {"refund_confirm_text"}),
    ("loop",         {"loop"},         True,  {"status_only"}),
    ("wrong_arg",    {"wrong_arg"},    True,  {"refund_eligible"}),
    ("injectable",   {"injectable"},   True,  {"prompt_injection"}),
    ("flaky_email",  {"flaky_email"},  True,  {"refund_and_email"}),
    # benign: must NOT raise a quality regression
    ("rephrase",     {"rephrase"},     False, set()),
    ("verbose(eff)", {"verbose"},      False, set()),
]


def run_bench(scenarios, trials=5):
    base = run_suite(MockAgent("good"), scenarios, trials, label="bench-base")
    rows = []
    for name, defects, should, expect in SEEDED:
        cand = run_suite(MockAgent("cand-" + name, defects), scenarios, trials, label=name)
        cmp = compare(base, cand)
        flagged = {q["scenario"] for q in cmp["quality_regressions"]}
        rows.append({"name": name, "should_flag": should, "flagged": cmp["flagged"],
                     "implicated": sorted(flagged), "located": expect <= flagged if should else None,
                     "eff_flags": sorted({e["metric"] for e in cmp["efficiency_regressions"]})})
    pos = [r for r in rows if r["should_flag"]]; neg = [r for r in rows if not r["should_flag"]]
    return {"trials": trials, "rows": rows, "caught": sum(r["flagged"] for r in pos), "seeded": len(pos),
            "located": sum(bool(r["located"]) for r in pos), "false_alarms": sum(r["flagged"] for r in neg),
            "benign": len(neg), "eff_caught": sum("cost" in r["eff_flags"] or "latency" in r["eff_flags"]
                                                   for r in rows if r["name"].startswith("verbose"))}
