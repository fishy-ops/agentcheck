from __future__ import annotations
import argparse, json, sys
from .core import ROOT, Scenario
from .agents import MockAgent, DEFECTS
from . import runner, agree as ag
from .compare import compare
from .bench import run_bench


def fmt(x, p=3):
    return "n/a" if x is None else f"{x:.{p}f}"


def agent_from(a):
    if a.agent == "mock":
        return MockAgent(a.label, set(filter(None, (a.defects or "").split(","))))
    from .agents import ClaudeAgent
    return ClaudeAgent(a.model, system=open(a.system_file).read() if a.system_file else "")


def print_summary(run):
    s = run["summary"]
    print(f"[{run['label']}] runs={s['runs']} pass={s['pass_rate']:.0%} score={s['mean_score']:.2f} "
          f"latency p50/p95={fmt(s['latency_p50_s'],2)}/{fmt(s['latency_p95_s'],2)}s "
          f"cost/run=${fmt(s['cost_per_run_usd'],5)} tokens/run={s['tokens_per_run']:.0f}"
          + ("  (SIMULATED mock numbers)" if s["simulated"] else ""))


def main(argv=None):
    p = argparse.ArgumentParser(prog="agentreg")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("label"); r.add_argument("--agent", default="mock", choices=["mock", "claude"])
    r.add_argument("--model"); r.add_argument("--system-file"); r.add_argument("--defects", help=",".join(sorted(DEFECTS)))
    r.add_argument("--trials", type=int, default=3); r.add_argument("--scenarios", default=str(ROOT / "scenarios"))
    c = sub.add_parser("compare"); c.add_argument("base"); c.add_argument("cand")
    l = sub.add_parser("label"); l.add_argument("run")
    g = sub.add_parser("agree"); g.add_argument("run")
    b = sub.add_parser("bench"); b.add_argument("--trials", type=int, default=5)
    a = p.parse_args(argv)

    if a.cmd == "run":
        scs = Scenario.load_dir(a.scenarios)
        run = runner.run_suite(agent_from(a), scs, a.trials, label=a.label)
        runner.save(run); print_summary(run)
        for sid, v in runner.per_scenario(run).items():
            print(f"  {'PASS' if v['pass_rate']==1 else 'FAIL' if v['pass_rate']==0 else 'FLAK'} {sid:22s} {v['pass_rate']:.0%} {v['failed_checks'] or ''}")
    elif a.cmd == "compare":
        base, cand = runner.load(a.base), runner.load(a.cand)
        print_summary(base); print_summary(cand)
        res = compare(base, cand)
        print(f"\nQUALITY REGRESSIONS: {len(res['quality_regressions'])}")
        for q in res["quality_regressions"]:
            print(f"  - {q['scenario']}: {q['base']:.0%} -> {q['cand']:.0%}  failing: {q['failed_checks']}")
        for e in res["efficiency_regressions"]:
            print(f"  ~ {e['scenario']} {e['metric']} +{e['pct']:.0%}")
        if res["improvements"]: print("  improved:", ", ".join(res["improvements"]))
        sys.exit(1 if res["flagged"] else 0)
    elif a.cmd == "label":
        run = runner.load(a.run); labels = ag.load_labels(a.run)
        for x in run["results"]:
            t = x["trace"]
            if t["trial"] != 0 or t["scenario_id"] in labels: continue
            print(f"\n=== {t['scenario_id']} ===\ncalls: {[(c['tool'], c['args']) for c in t['tool_calls']]}\nfinal: {t['final']}")
            v = input("your verdict [p]ass/[f]ail/[s]kip/[q]uit: ").strip().lower()
            if v == "q": break
            if v in ("p", "f"): labels[t["scenario_id"]] = "pass" if v == "p" else "fail"
            ag.labels_path(a.run).write_text(json.dumps(labels, indent=1))
    elif a.cmd == "agree":
        res = ag.agreement(runner.load(a.run), ag.load_labels(a.run))
        if not res["n"]: print("no manual labels yet; run: agentreg label", a.run); return
        print(f"n={res['n']} agreement={res['agreement']:.0%} kappa={res['kappa']:.2f} "
              f"auto-too-lenient={res['too_lenient']} auto-too-strict={res['too_strict']}")
        for d in res["disagreements"]: print("  ", d)
    elif a.cmd == "bench":
        res = run_bench(Scenario.load_dir(ROOT / "scenarios"), a.trials)
        for x in res["rows"]:
            print(f"  {x['name']:14s} expect={'FLAG' if x['should_flag'] else 'quiet'} got={'FLAG' if x['flagged'] else 'quiet'} "
                  f"implicated={x['implicated']} eff={x['eff_flags']}")
        print(f"\nRegressions caught: {res['caught']}/{res['seeded']}  (localized to right scenario: {res['located']}/{res['seeded']})"
              f"\nFalse alarms on benign changes: {res['false_alarms']}/{res['benign']}"
              f"\nEfficiency regression (verbose) caught: {res['eff_caught']}/1")
        (ROOT / "runs" / "bench.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
