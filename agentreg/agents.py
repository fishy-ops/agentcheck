"""Agent adapters. An agent is a callable: (scenario, trial) -> Trace."""
from __future__ import annotations
import random, re, time
from .core import Scenario, ToolEnv, Trace

# Known defects the mock can inject. Used for the seeded-regression benchmark.
DEFECTS = {"skip_lookup", "over_refund", "no_confirm", "loop", "wrong_arg",
           "injectable", "flaky_email", "verbose", "rephrase"}


class MockAgent:
    """Deterministic stand-in for a prompt/model version. `defects` = set of DEFECTS."""
    def __init__(self, version="good", defects=(), seed=0):
        self.version, self.defects, self.seed = version, set(defects), seed

    def __call__(self, sc: Scenario, trial: int) -> Trace:
        rng = random.Random(f"{self.seed}-{sc.id}-{trial}-{self.version}")
        env, d, u = ToolEnv(sc), self.defects, sc.user
        steps, calls = 1, []

        def call(name, args):
            nonlocal steps
            steps += 1
            r = env.call(name, args); calls.append({"tool": name, "args": args, "result": r}); return r

        oid = (re.search(r"\b([A-Z]\d{3})\b", u) or [None, None])[1]
        email = (re.search(r"[\w.]+@[\w.]*\w", u) or [None])[0]
        low = u.lower()
        wants_refund = "refund" in low
        final = ""
        if "policy" in low:
            r = call("search_kb", {"query": "return policy"})
            final = f"Our return policy: {r.get('answer', 'unavailable')}"
        elif oid:
            order = None
            if "skip_lookup" not in d:
                order = call("lookup_order", {"order_id": oid})
                if "loop" in d:
                    for _ in range(3):
                        call("lookup_order", {"order_id": oid})
            if order and order.get("error"):
                final = f"Sorry, I couldn't find order {oid}."
            elif wants_refund:
                days = (order or {}).get("days_since_delivery", 0)
                amt = (order or {}).get("total")
                eligible = "over_refund" in d or (order is not None and days <= 30)
                if eligible:
                    rid = oid if "wrong_arg" not in d else oid[0] + oid[1:][::-1]
                    call("issue_refund", {"order_id": rid, "amount": amt})
                    if email and not ("flaky_email" in d and rng.random() < 0.6):
                        call("send_email", {"to": email, "subject": f"Refund for {oid}"})
                    final = (f"I've refunded order {oid}." if "no_confirm" in d
                             else f"I've refunded ${amt:.2f} for order {oid}." if amt is not None
                             else f"I've refunded order {oid}.")
                else:
                    final = f"Order {oid} is outside the 30-day refund window, so I can't refund it."
            else:
                note = (order or {}).get("notes", "")
                if "injectable" in d and "refund" in note.lower():
                    call("issue_refund", {"order_id": oid, "amount": 500})
                final = f"Order {oid} is {(order or {}).get('status', 'unknown')}."
        else:
            final = "Could you share your order number?"
        if "rephrase" in d:
            final = final.replace("I've refunded", "Done, we have refunded").replace("Sorry, I couldn't find", "Hmm, no record of")
        out_tok = max(20, len(final) // 4) + 35 * len(calls)
        in_tok = 300 + 220 * steps
        if "verbose" in d:
            out_tok *= 3; in_tok = int(in_tok * 1.8)
        lat = 0.5 + 0.9 * steps + rng.uniform(0, 0.4) + (1.5 if "verbose" in d else 0)
        return Trace(sc.id, trial, calls, final, steps, in_tok, out_tok, round(lat, 3), "mock", simulated=True)


class ClaudeAgent:
    """Real tool-using agent against the scenario's canned tools (needs ANTHROPIC_API_KEY)."""
    def __init__(self, model: str, system: str = "", max_turns: int = 8):
        import anthropic
        self.c, self.model, self.system, self.max_turns = anthropic.Anthropic(), model, system, max_turns

    def __call__(self, sc: Scenario, trial: int) -> Trace:
        env = ToolEnv(sc)
        tools = [{"name": n, "description": s.get("description", n),
                  "input_schema": s.get("input_schema", {"type": "object", "properties": {}})}
                 for n, s in sc.tools.items()]
        msgs = [{"role": "user", "content": sc.user}]
        calls, tin, tout, steps, final, err = [], 0, 0, 0, "", None
        t0 = time.perf_counter()
        try:
            for _ in range(self.max_turns):
                r = self.c.messages.create(model=self.model, max_tokens=1024,
                                           system=self.system or sc.system or "You are a support agent.",
                                           tools=tools, messages=msgs)
                steps += 1; tin += r.usage.input_tokens; tout += r.usage.output_tokens
                msgs.append({"role": "assistant", "content": r.content})
                uses = [b for b in r.content if b.type == "tool_use"]
                if not uses:
                    final = "".join(b.text for b in r.content if b.type == "text"); break
                results = []
                for b in uses:
                    res = env.call(b.name, dict(b.input)); calls.append({"tool": b.name, "args": dict(b.input), "result": res})
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": str(res)})
                msgs.append({"role": "user", "content": results})
            else:
                err = "max_turns exceeded"
        except Exception as e:  # recorded, scored as failure
            err = f"{type(e).__name__}: {e}"
        return Trace(sc.id, trial, calls, final, steps, tin, tout, round(time.perf_counter() - t0, 3),
                     self.model, error=err)
