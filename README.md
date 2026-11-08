# agentcheck

Run any tool-using agent against saved scenarios, score each run, and flag regressions between a baseline and a candidate (new prompt or model).

```
python3 -m agentreg run baseline  --agent claude --model <id> --system-file prompt_v1.txt --trials 5
python3 -m agentreg run candidate --agent claude --model <id> --system-file prompt_v2.txt --trials 5
python3 -m agentreg compare baseline candidate     # exit 1 if a quality regression is flagged
python3 -m agentreg label baseline                 # record YOUR pass/fail verdicts
python3 -m agentreg agree baseline                 # agreement % + Cohen's kappa vs. your labels
python3 -m agentreg bench                          # seeded-defect benchmark (mock agent)
```

Scenarios: `scenarios/*.json` hold the user message, canned tool responses, and checks (`tool_called`, `tool_not_called`, `tool_before`, `final_contains`, `final_not_contains`, `max_tool_calls`, `max_steps`). A run passes when every required check passes. Add real model prices to `pricing.json` (`in_per_mtok`, `out_per_mtok`) or cost shows as n/a.

## The three measurements
- **Regressions caught**: `bench` injects 7 known defects plus 2 benign changes into the mock agent. It reports caught/seeded, whether each is localized to the right scenario, and false alarms. A regression means the pass rate drops by at least 34 points across trials. Cost (+25%) and latency (+50%) are flagged separately as efficiency regressions.
- **Agreement with your manual checks**: `label` then `agree`. Reports agreement, kappa, and whether the auto-scorer is too lenient or too strict, with the disagreeing checks listed. No labels are shipped, because they have to be yours.
- **Cost and latency per run**: every summary line shows p50/p95 latency, cost per run and tokens per run. Mock-agent numbers are simulated and labelled as such. Claude runs use measured wall-clock time and API token usage.
