#!/usr/bin/env python3
"""
Bayu's FutureEval forecasting bot (Metaculus Fall 2026 Bot Tournament + MiniBench).

Pipeline: fetch open questions -> reason (LLM via OpenRouter, with a no-LLM
fallback) -> build API payload -> (only with --submit) post forecast + the
required private reasoning comment.

Design notes (from the official Metaculus template bot,
github.com/Metaculus/metac-bot-template):
  * API base: https://www.metaculus.com/api
  * Auth header: "Authorization: Token <METACULUS_TOKEN>" — as of Fall 2026 the
    API requires a token even for reads.
  * Submit:  POST /api/questions/forecast/  [{"question": id, "source": "api", ...}]
  * Comment: POST /api/comments/create/ {text, included_forecast: true,
             is_private: true, on_post: post_id}  (comments are REQUIRED for prizes)
  * Tournaments: Fall 2026 FutureEval = 33121 / "fall-futureeval-2026",
    MiniBench = "minibench", testing = "bot-testing-area".

Safety: dry-run is the DEFAULT. Nothing is posted unless --submit is passed
AND METACULUS_TOKEN is set. Stdlib only (urllib) — no dependencies.

Secrets: read from environment only. OPENROUTER_API_KEY may also be loaded by
the wrapper (run.sh) from ~/.openrouter/key into the process environment.
Keys are never printed or written to disk by this program.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import http.client
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

API_BASE_URL = "https://www.metaculus.com/api"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

FALL_2026_ID = 33121
DEFAULT_TOURNAMENTS = ["fall-futureeval-2026", "minibench"]
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecasted.json")


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------

def _request(method: str, url: str, *, token: str | None = None,
             json_body: dict | list | None = None,
             extra_headers: dict | None = None, timeout: int = 60) -> dict:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json",
               "Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = f"Token {token}"
    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    last_err: Exception | None = None
    for _attempt in range(2):  # one retry: some proxies truncate chunked reads
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                try:
                    raw = resp.read().decode("utf-8", "replace")
                except http.client.IncompleteRead as e:
                    # The body often arrives intact; only the terminating
                    # chunk is lost in transit. Parse what we received.
                    raw = e.partial.decode("utf-8", "replace")
            return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"HTTP {e.code} for {url}: {body}") from None
        except (http.client.IncompleteRead, json.JSONDecodeError,
                ConnectionError, TimeoutError) as e:
            last_err = e
    raise RuntimeError(f"request to {url} failed after retry: {last_err}")


def api_get(path: str, params: dict, token: str | None) -> dict:
    qs = urllib.parse.urlencode(params, doseq=True)
    return _request("GET", f"{API_BASE_URL}{path}?{qs}", token=token)


# --------------------------------------------------------------------------
# Question fetching / normalisation
# --------------------------------------------------------------------------

def _community_prediction(question: dict) -> float | None:
    """Best-effort extraction of the community prediction for a binary question
    from the various shapes the v2 API has used over time."""
    try:
        agg = question.get("aggregations") or {}
        for method in ("recency_weighted", "unweighted", "metaculus_prediction"):
            latest = (agg.get(method) or {}).get("latest") or {}
            for field in ("centers", "means", "forecast_values"):
                vals = latest.get(field)
                if vals:
                    return float(vals[0])
        cp = question.get("question_prediction")
        if cp is not None:
            return float(cp)
    except (TypeError, ValueError, IndexError):
        return None
    return None


def normalize_question(post: dict) -> dict | None:
    q = post.get("question")
    if not q or q.get("status") not in (None, "open"):
        return None
    return {
        "question_id": q.get("id"),
        "post_id": post.get("id"),
        "type": q.get("type"),
        "title": q.get("title") or "",
        "description": q.get("description") or "",
        "resolution_criteria": q.get("resolution_criteria") or "",
        "fine_print": q.get("fine_print") or "",
        "options": q.get("options") or [],
        "scheduled_close_time": q.get("scheduled_close_time"),
        "community_prediction": _community_prediction(q),
        # numeric scaffolding (may be absent depending on API shape)
        "scaling": q.get("scaling") or {},
        "open_upper_bound": q.get("open_upper_bound"),
        "open_lower_bound": q.get("open_lower_bound"),
        "unit": q.get("unit"),
    }


def fetch_open_questions(tournament: str | int, token: str | None,
                         limit: int = 50) -> list[dict]:
    """Fetch open forecastable posts from a tournament. The live API requires
    a token even for reads; without one it returns a permission error, which
    we surface verbatim."""
    data = api_get(
        "/posts/",
        {
            "limit": limit,
            "offset": 0,
            "order_by": "-hotness",
            "forecast_type": "binary,multiple_choice,numeric,discrete",
            "tournaments": tournament,
            "statuses": "open",
            "include_description": "true",
        },
        token,
    )
    out = []
    for post in data.get("results", []):
        nq = normalize_question(post)
        if nq and nq["type"] in ("binary", "multiple_choice", "numeric", "discrete"):
            out.append(nq)
    return out


# --------------------------------------------------------------------------
# No-LLM fallback forecaster (base rate + community-prediction blend)
# --------------------------------------------------------------------------

def _clamp(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return min(hi, max(lo, p))


def fallback_binary(q: dict) -> tuple[float, str]:
    cp = q.get("community_prediction")
    prior = 0.5  # uninformed base rate for a binary proposition
    if cp is not None:
        p = 0.7 * cp + 0.3 * prior
        reasoning = (
            f"Base rate: with no question-specific evidence, the reference-class "
            f"prior for a binary Metaculus question is 50%. Community prediction "
            f"is {cp:.0%}, which aggregates many forecasters' evidence; the "
            f"fallback blends it 70/30 with the prior -> {_clamp(p):.0%}. "
            f"(No-LLM fallback: no independent research was performed.)"
        )
    else:
        p = prior
        reasoning = (
            "Base rate: no community prediction was available and no LLM "
            "research ran, so the fallback stays at the uninformed 50% prior "
            "rather than inventing evidence. (No-LLM fallback.)"
        )
    return _clamp(p), reasoning


def fallback_multiple_choice(q: dict) -> tuple[dict[str, float], str]:
    options = q["options"]
    n = len(options)
    uniform = 1.0 / n
    probs = {opt: uniform for opt in options}
    reasoning = (
        f"Base rate: {n} options and no usable per-option evidence in "
        f"fallback mode, so each option gets the uniform {uniform:.0%}. "
        f"(No-LLM fallback: deliberately uninformative.)"
    )
    return probs, reasoning


def fallback_numeric_percentiles(q: dict) -> tuple[dict[int, float], str]:
    scaling = q.get("scaling") or {}
    lo = scaling.get("range_min")
    hi = scaling.get("range_max")
    if lo is None or hi is None:
        raise ValueError("numeric question without range bounds")
    mid = (lo + hi) / 2.0
    spread = (hi - lo)
    pct = {
        10: lo + 0.30 * spread, 20: lo + 0.38 * spread, 40: lo + 0.46 * spread,
        60: lo + 0.54 * spread, 80: lo + 0.62 * spread, 90: lo + 0.70 * spread,
    }
    reasoning = (
        f"Base rate: fallback assumes a wide distribution centred on the "
        f"range midpoint ({mid}) between the bounds {lo} and {hi}, with "
        f"percentiles spread across the central 40% of the range. "
        f"(No-LLM fallback: no independent research was performed.)"
    )
    return pct, reasoning


# --------------------------------------------------------------------------
# Numeric CDF construction (percentiles -> 201-point CDF, Metaculus format)
# --------------------------------------------------------------------------

def percentiles_to_cdf(percentiles: dict[int, float], lo: float, hi: float,
                       size: int = 201) -> list[float]:
    pts = sorted((p / 100.0, v) for p, v in percentiles.items())
    # value at percentile y via linear interpolation, flat extrapolation
    def value_at(y: float) -> float:
        if y <= pts[0][0]:
            return pts[0][1]
        if y >= pts[-1][0]:
            return pts[-1][1]
        for (y0, v0), (y1, v1) in zip(pts, pts[1:]):
            if y0 <= y <= y1:
                if y1 == y0:
                    return v1
                return v0 + (v1 - v0) * (y - y0) / (y1 - y0)
        return pts[-1][1]

    cdf = []
    prev = 0.0
    for i in range(size):
        x = lo + (hi - lo) * i / (size - 1)
        # invert: find y such that value_at(y) == x, by scanning the segments
        y = _invert(value_at, x, pts)
        # enforce Metaculus standardization: strictly increasing, min step,
        # and no single step above 0.2 of mass
        y = max(y, prev + (0.01 / (size - 1)) if i > 0 else y)
        y = min(y, prev + 0.19) if i > 0 else y
        y = min(max(y, 0.0), 1.0)
        cdf.append(y)
        prev = y
    # final point must be exactly 1.0 for closed upper bound handling; keep
    # monotonic after forcing
    cdf[-1] = 1.0
    for i in range(size - 2, -1, -1):
        if cdf[i] > cdf[i + 1]:
            cdf[i] = cdf[i + 1]
    return cdf


def _invert(value_at, x: float, pts) -> float:
    lo_y, hi_y = 0.0, 1.0
    for _ in range(40):
        mid = (lo_y + hi_y) / 2
        if value_at(mid) < x:
            lo_y = mid
        else:
            hi_y = mid
    return (lo_y + hi_y) / 2


# --------------------------------------------------------------------------
# LLM forecaster (OpenRouter)
# --------------------------------------------------------------------------

BINARY_PROMPT = """You are a professional forecaster. Reason like a superforecaster.

Question: {title}

Background:
{background}

Resolution criteria:
{resolution_criteria}
{fine_print}

{community_line}
Today is {today}. The question closes at {close_time}.

Write your reasoning in this order:
(a) Base rate: how often do things like this happen, before case specifics?
(b) The strongest evidence FOR a Yes outcome.
(c) The strongest evidence AGAINST (for a No outcome).
(d) How the community prediction (if given) should be weighed, and where you disagree with it if you do.
Then finish with exactly one final line: "Probability: ZZ%" with ZZ an integer 1-99.
"""

MC_PROMPT = """You are a professional forecaster. Reason like a superforecaster.

Question: {title}

Options: {options}

Background:
{background}

Resolution criteria:
{resolution_criteria}
{fine_print}

Today is {today}. The question closes at {close_time}.

Write your reasoning in this order:
(a) Base rate for each option before case specifics.
(b) The key evidence that discriminates between the options.
(c) Why the remaining options still keep some probability.
Then finish with one line per option, in the exact order given, formatted as:
Option_A: NN%
Option_B: NN%
... (percentages roughly summing to 100)
"""

NUMERIC_PROMPT = """You are a professional forecaster. Reason like a superforecaster.

Question: {title}

Background:
{background}

Resolution criteria:
{resolution_criteria}
{fine_print}

Units: {unit}. The outcome must lie between {lo} and {hi}.

Today is {today}. The question closes at {close_time}.

Write your reasoning in this order:
(a) The outcome if nothing changed (status quo anchor).
(b) The outcome if the current trend continued.
(c) A low scenario and a high scenario.
Then finish with exactly these lines (plain numbers, no commas, no scientific notation):
Percentile 10: X
Percentile 20: X
Percentile 40: X
Percentile 60: X
Percentile 80: X
Percentile 90: X
"""


def _openrouter_via_curl(body: dict, api_key: str) -> dict:
    """Transport fallback: some proxies truncate urllib's chunked reads of
    OpenRouter responses; curl handles them reliably. The request body goes
    via stdin; the key travels in a header argument (local, single-user host).
    """
    import subprocess
    proc = subprocess.run(
        ["curl", "-sS", "--max-time", "240", "-X", "POST", OPENROUTER_URL,
         "-H", f"Authorization: Bearer {api_key}",
         "-H", "Content-Type: application/json",
         "-d", "@-"],
        input=json.dumps(body).encode("utf-8"),
        capture_output=True, timeout=260,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"curl transport failed: {proc.stderr.decode()[:200]}")
    return json.loads(proc.stdout.decode("utf-8", "replace"))


def call_openrouter(prompt: str, api_key: str, model: str) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": int(os.environ.get("OPENROUTER_MAX_TOKENS", "2500")),
    }
    try:
        data = _request(
            "POST", OPENROUTER_URL, json_body=body,
            extra_headers={"Authorization": f"Bearer {api_key}"},
        )
    except Exception:
        data = _openrouter_via_curl(body, api_key)
    if "choices" not in data:
        err = (data.get("error") or {}).get("message")
        if not err:
            err = json.dumps(data)[:200]
        raise ValueError(f"OpenRouter error: {err}")
    msg = data["choices"][0]["message"]
    content = msg.get("content")
    if not content:
        # Reasoning models may leave content empty and put everything in
        # `reasoning`; the required final line lives there in that case.
        content = msg.get("reasoning")
    if not content:
        raise ValueError("empty LLM reply (no content or reasoning)")
    return content


def _today() -> str:
    return _dt.date.today().isoformat()


def llm_binary(q: dict, api_key: str, model: str) -> tuple[float, str]:
    cp = q.get("community_prediction")
    community_line = (
        f"The current community prediction is {cp:.0%}." if cp is not None
        else "No community prediction is available yet."
    )
    prompt = BINARY_PROMPT.format(
        title=q["title"], background=q["description"][:6000],
        resolution_criteria=q["resolution_criteria"][:3000],
        fine_print=q["fine_print"][:1000], community_line=community_line,
        today=_today(), close_time=q.get("scheduled_close_time"),
    )
    text = call_openrouter(prompt, api_key, model)
    matches = re.findall(r"(\d{1,3})\s*%", text)
    if not matches:
        raise ValueError("no percentage in LLM reply")
    p = _clamp(int(matches[-1]) / 100.0)
    return p, text.strip()


def llm_multiple_choice(q: dict, api_key: str, model: str) -> tuple[dict[str, float], str]:
    prompt = MC_PROMPT.format(
        title=q["title"], options=q["options"],
        background=q["description"][:6000],
        resolution_criteria=q["resolution_criteria"][:3000],
        fine_print=q["fine_print"][:1000],
        today=_today(), close_time=q.get("scheduled_close_time"),
    )
    text = call_openrouter(prompt, api_key, model)
    nums = []
    for line in text.splitlines():
        m = re.match(r"\s*Option[\s_]?[A-Z]\s*:\s*(\d{1,3})\s*%", line, re.I)
        if m:
            nums.append(int(m.group(1)))
    if len(nums) != len(q["options"]):
        # Second chance (same trick the official template uses): take the
        # last N percentages in the reply, in order.
        all_pct = [int(x) for x in re.findall(r"(\d{1,3})\s*%", text)]
        if len(all_pct) >= len(q["options"]):
            nums = all_pct[-len(q["options"]):]
    if len(nums) != len(q["options"]):
        raise ValueError(f"expected {len(q['options'])} option lines, got {len(nums)}")
    total = sum(nums) or 1
    probs = {}
    for opt, n in zip(q["options"], nums):
        probs[opt] = _clamp(n / total, 0.01, 0.97)
    s = sum(probs.values())
    probs = {k: v / s for k, v in probs.items()}
    return probs, text.strip()


def llm_numeric(q: dict, api_key: str, model: str) -> tuple[dict[int, float], str]:
    scaling = q.get("scaling") or {}
    prompt = NUMERIC_PROMPT.format(
        title=q["title"], background=q["description"][:6000],
        resolution_criteria=q["resolution_criteria"][:3000],
        fine_print=q["fine_print"][:1000], unit=q.get("unit") or "as stated",
        lo=scaling.get("range_min"), hi=scaling.get("range_max"),
        today=_today(), close_time=q.get("scheduled_close_time"),
    )
    text = call_openrouter(prompt, api_key, model)
    pct: dict[int, float] = {}
    for line in text.splitlines():
        m = re.match(r"\s*Percentile\s+(\d+)\s*:\s*(-?[\d.]+)", line)
        if m:
            pct[int(m.group(1))] = float(m.group(2))
    if len(pct) < 4:
        raise ValueError(f"too few percentiles parsed: {pct}")
    vals = [pct[k] for k in sorted(pct)]
    if any(b <= a for a, b in zip(vals, vals[1:])):
        raise ValueError("percentiles not strictly increasing")
    return pct, text.strip()


# --------------------------------------------------------------------------
# Forecast orchestration
# --------------------------------------------------------------------------

def forecast_question(q: dict, api_key: str | None, model: str) -> dict:
    """Returns {forecast, payload, reasoning, method} for one question."""
    qtype = q["type"]
    method = "fallback"
    reasoning = ""
    forecast = None
    percentiles = None

    if api_key:
        try:
            if qtype == "binary":
                forecast, reasoning = llm_binary(q, api_key, model)
            elif qtype == "multiple_choice":
                forecast, reasoning = llm_multiple_choice(q, api_key, model)
            elif qtype in ("numeric", "discrete"):
                percentiles, reasoning = llm_numeric(q, api_key, model)
            method = f"llm:{model}"
        except Exception as e:  # any LLM failure -> deterministic fallback
            reasoning = f"[LLM path failed ({e}); fallback used]\n"
            method = "fallback"
            forecast = None
            percentiles = None

    if method == "fallback":
        if qtype == "binary":
            forecast, r = fallback_binary(q)
            reasoning += r
        elif qtype == "multiple_choice":
            forecast, r = fallback_multiple_choice(q)
            reasoning += r
        elif qtype in ("numeric", "discrete"):
            percentiles, r = fallback_numeric_percentiles(q)
            reasoning += r

    if qtype in ("numeric", "discrete"):
        scaling = q.get("scaling") or {}
        size = 201
        if qtype == "discrete" and scaling.get("inbound_outcome_count"):
            size = int(scaling["inbound_outcome_count"]) + 1
        cdf = percentiles_to_cdf(percentiles, float(scaling["range_min"]),
                                 float(scaling["range_max"]), size=size)
        payload = {"probability_yes": None, "probability_yes_per_category": None,
                   "continuous_cdf": cdf}
        forecast_repr = {"percentiles": percentiles}
    elif qtype == "binary":
        payload = {"probability_yes": forecast, "probability_yes_per_category": None,
                   "continuous_cdf": None}
        forecast_repr = forecast
    else:  # multiple_choice
        payload = {"probability_yes": None, "probability_yes_per_category": forecast,
                   "continuous_cdf": None}
        forecast_repr = forecast

    return {"forecast": forecast_repr, "payload": payload,
            "reasoning": reasoning, "method": method}


# --------------------------------------------------------------------------
# Submission (only reachable with --submit AND a token)
# --------------------------------------------------------------------------

def submit_forecast(q: dict, result: dict, token: str) -> None:
    body = [{"question": q["question_id"], "source": "api", **result["payload"]}]
    _request("POST", f"{API_BASE_URL}/questions/forecast/",
             token=token, json_body=body)
    # Comments are required to be prize-eligible: post the reasoning privately.
    comment = (
        f"Forecast by an automated bot (FutureEval Fall 2026).\n\n"
        f"Method: {result['method']}\n\n{result['reasoning']}"
    )[:9000]
    _request("POST", f"{API_BASE_URL}/comments/create/", token=token, json_body={
        "text": comment, "parent": None, "included_forecast": True,
        "is_private": True, "on_post": q["post_id"],
    })


def load_state() -> set:
    try:
        with open(STATE_FILE) as f:
            return set(json.load(f))
    except (OSError, ValueError):
        return set()


def save_state(ids: set) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(sorted(ids), f)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def format_forecast_line(q: dict, result: dict) -> str:
    f = result["forecast"]
    if q["type"] == "binary":
        return f"P(Yes) = {f:.0%}"
    if q["type"] == "multiple_choice":
        return ", ".join(f"{k}: {v:.0%}" for k, v in f.items())
    return "percentiles: " + ", ".join(
        f"P{k}={v}" for k, v in f["percentiles"].items())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Metaculus FutureEval bot")
    ap.add_argument("--tournaments", nargs="+", default=DEFAULT_TOURNAMENTS,
                    help="tournament ids/slugs to poll")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--submit", action="store_true",
                    help="actually post forecasts (default: dry-run)")
    ap.add_argument("--questions-file",
                    help="offline JSON list of normalized questions (testing)")
    ap.add_argument("--out", help="write a markdown report of this run here")
    args = ap.parse_args(argv)

    token = os.environ.get("METACULUS_TOKEN") or None
    or_key = os.environ.get("OPENROUTER_API_KEY") or None
    # Free slugs rotate; this one was verified working 2026-11-10 (a slow
    # reasoning model — fine for samples; for live MiniBench volume use the
    # Metaculus sponsored credits and set OPENROUTER_MODEL to a faster model,
    # e.g. google/gemma-4-31b-it:free when its upstream limit is clear).
    model = os.environ.get(
        "OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

    if args.submit and not token:
        print("ERROR: --submit requires METACULUS_TOKEN in the environment.",
              file=sys.stderr)
        return 2

    questions: list[dict] = []
    if args.questions_file:
        with open(args.questions_file) as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} questions from {args.questions_file}")
    else:
        for t in args.tournaments:
            try:
                qs = fetch_open_questions(t, token, limit=args.limit)
            except RuntimeError as e:
                print(f"[{t}] fetch failed: {e}")
                continue
            print(f"[{t}] {len(qs)} open questions")
            questions.extend(qs)

    done = load_state()
    report = [f"# Bot run — {_dt.datetime.now().isoformat(timespec='seconds')}",
              f"Mode: {'SUBMIT' if args.submit else 'DRY-RUN'} · "
              f"LLM: {'on (' + model + ')' if or_key else 'off (fallback only)'}", ""]

    submitted = 0
    for q in questions[: args.limit]:
        if q.get("question_id") in done:
            print(f"skip (already forecasted): {q['title'][:70]}")
            continue
        result = forecast_question(q, or_key, model)
        line = format_forecast_line(q, result)
        print(f"- [{q['type']}] {q['title'][:80]} -> {line} ({result['method']})")
        report += [f"## {q['title']}", "",
                   f"- Type: {q['type']} · Method: {result['method']}",
                   f"- Forecast: {line}",
                   f"- Community prediction: "
                   f"{('%.0f%%' % (q['community_prediction'] * 100)) if q.get('community_prediction') is not None else 'n/a'}",
                   "", "**Reasoning:**", "", result["reasoning"], ""]
        if args.submit:
            submit_forecast(q, result, token)  # guarded above: token required
            if q.get("question_id") is not None:
                done.add(q["question_id"])
            submitted += 1

    if args.submit:
        save_state(done)
    print(f"Done. {'Submitted' if args.submit else 'Dry-ran'} "
          f"{submitted if args.submit else len(questions)} forecasts.")
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(report))
        print(f"Report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
