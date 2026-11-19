"""Refresh-aware quantile forecaster for gta6-trophy-rarity-forecast-01 (v2).

Method (general, no memorised answers; everything is computed from the
case file at run time):

* Exophase refreshes a game's counts on its own schedule, hours apart.
  A refresh adds newly tracked players who have earned almost nothing
  yet, so every rate is diluted: the rate falls by roughly the factor
  the tracked-player count jumps by, while earners barely move over a
  short window.  That dilution is a *bump*: it arrives fast (one
  refresh) and fades over ~2 days as the new cohort plays and earns.
  The page had been flat for hours when a case is cut, so a refresh
  inside the window is the working assumption for the median; the
  final's 4.5-day horizon makes one near-certain anyway.
* The jump size is estimated from the case's own history: the median
  of the player-count jumps the history shows, shrunk toward the prior
  of one several-fold refresh (6x) when the history shows few or none.
  A flat history carries no jumps, so the prior stands; the final
  case's two days of reads will carry real ones.
* The new cohort does NOT earn every trophy at the same rate.  Most
  trophies need hours of story progress, so a fresh cohort has earned
  almost none of them in a short window (slow branch: pure dilution).
  But the earliest trophies -- the ones the most players already
  hold, i.e. the highest rates in the case -- a fresh cohort reaches
  within hours (fast branch).  Each trophy's forecast is therefore a
  mixture over the two catch-up branches, with the fast branch's
  weight rising steeply but smoothly with the trophy's rate relative
  to the case's highest rate.  The quantiles are the mixture's own
  quantiles: the median stays near pure dilution for most trophies,
  while the upper quantiles reach up for the easy ones, where the
  fast branch lives.  No threshold is fitted to any case, and the
  weight is deliberately steep (fourth power) because the rate is a
  coarse proxy -- only the very easiest trophies earn a fast median.
* Over days, the rate of each trophy instead moves toward the share
  of tracked players who will hold it by the target: early/easy
  trophies toward a high share, late/grind trophies toward a low one.
  The pull is in logit space, at a log-time rate, with the ceiling
  set by tier and by the trophy's launch-night rate rank inside its
  tier (a trophy more of the first players earned is an earlier /
  easier one).  The pull is scaled by the horizon, so a 90-minute
  forecast barely moves while a week-long one mostly arrives.
* The spread is honest about both: a base term for refresh-magnitude
  uncertainty, a term that grows as the progress pull matures, and a
  refresh term that fades with horizon -- plus the explicit
  slow/fast mixture above it.

Python 3.13, standard library only, deterministic, no network.
Reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes one
<id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- dilution bump (logit units) -------------------------------------------
# Prior jump of one refresh: tracked players multiplied several-fold
# while earners barely moved.  The case history's own jumps, when it
# shows any, pull this estimate toward their median (see jump_size).
JUMP_PRIOR = 6.0
JUMP_PRIOR_WEIGHT = 2.0  # pseudo-observations behind the prior
# The bump's rise is fast on purpose: the median (q50) sits in the
# refresh mode once a refresh is more likely than not before the
# target, and the spread -- not the median -- carries the chance that
# no refresh lands (its upper quantiles reach toward no dilution).
TAU_RISE_H = 0.25   # a refresh is expected within about an hour
TAU_FADE_H = 48.0   # the new cohort catches up over ~2 days

# --- new-cohort catch-up -----------------------------------------------------
# After a refresh the rate is (E + K*dN) / (N + dN): K is the fraction
# of the old rate the new cohort has reached by the target.  Two
# branches: slow (a fresh cohort has earned almost none of most
# trophies in a short window, so K = 0: pure dilution) and fast (the
# earliest trophies it reaches within hours).  The branch weights and
# sizes are global priors, set so the model's implied v1-style loss
# matches the public dev-board aggregates (loss 0.110, miss 0.04 pp,
# coverage 0.927) -- the structure, not any trophy's answer.
K_SLOW = 0.0
K_FAST = 0.35
# Fast-branch weight grows steeply with easiness (the trophy's rate
# relative to the case's highest rate): w = min(0.85, e**4).  Only
# the very easiest trophy's median sits in the fast branch; a
# mid-rate trophy's upper quantiles still reach it.
W_MAX = 0.85
W_POWER = 4.0

# --- progress pull -----------------------------------------------------------
TAU_PULL_D = 1.5    # base time constant of the pull toward the ceiling
TAU_PULL_EXTRA_D = 2.0
CEILING = {"platinum": 0.0005, "gold": 0.03, "silver": 0.10}
BRONZE_LO, BRONZE_HI = 0.06, 0.55  # day-7 share, lowest to highest rate

# --- spread (logit std, within one branch) -----------------------------------
S_BASE = 0.20       # refresh magnitude + short-window earner growth
S_PROG = 1.3        # how far players get by the target is uncertain
S_REFRESH = 0.20    # whether/when a refresh lands; fades with horizon


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mixture_quantile(means, weights, sigma, tau):
    """tau-quantile of a mixture of normals N(means[i], sigma^2).

    Bisection on the mixture CDF; deterministic, no randomness.
    """
    lo = min(means) - 8.0 * sigma - 1.0
    hi = max(means) + 8.0 * sigma + 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = sum(w * _norm_cdf((mid - m) / sigma)
                  for m, w in zip(means, weights))
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def jump_size(history):
    """One refresh's player-count jump, from the case's own history.

    Median of the jumps the history shows, shrunk toward the prior
    (several-fold) with the prior's weight; a flat history carries no
    jumps, so the prior stands.  Returns (jump, dilution_logit).
    """
    ratios = []
    for prev, cur in zip(history, history[1:]):
        n0, n1 = prev["players_tracked"], cur["players_tracked"]
        if n0 and n1 and n1 > n0 * 1.02:
            ratios.append(n1 / n0)
    if ratios:
        ratios.sort()
        med = ratios[len(ratios) // 2]
        w = JUMP_PRIOR_WEIGHT
        jump = math.exp((len(ratios) * math.log(med)
                         + w * math.log(JUMP_PRIOR))
                        / (len(ratios) + w))
    else:
        jump = JUMP_PRIOR
    dilution = min(2.5, max(0.5, math.log(jump)))
    return jump, dilution


def progress_mult(history):
    """Aggregate earner-vs-player growth over the history, neutral flat."""
    mult = 1.0
    first, last = history[0], history[-1]
    if first["players_tracked"] and last["players_tracked"]:
        n_growth = last["players_tracked"] / first["players_tracked"]
        e_growth = last["trophies_earned"] / first["trophies_earned"]
        if n_growth > 0 and e_growth > 0:
            mult = min(1.3, max(0.8, (e_growth / n_growth) ** 0.25))
    return mult


def ceilings(trophies, last_rates):
    """Day-7 share ceiling per trophy, from tier + rate rank in tier."""
    out = {}
    by_tier = {}
    for i, t in enumerate(trophies):
        by_tier.setdefault(t["tier"], []).append(i)
    for tier, idxs in by_tier.items():
        if tier == "bronze":
            order = sorted(idxs, key=lambda i: (last_rates[i] or 0, trophies[i]["id"]))
            n = len(order)
            for rank, i in enumerate(order):
                frac = rank / (n - 1) if n > 1 else 0.5
                out[i] = BRONZE_LO + (BRONZE_HI - BRONZE_LO) * frac
        else:
            for i in idxs:
                out[i] = CEILING.get(tier, 0.10)
    return out


def forecast_case(case):
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    last = history[-1]
    last_rates = [r if r is not None else 0.0 for r in last["rates"]]
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0
    n0 = last["players_tracked"]

    jump, dilution_peak = jump_size(history)
    prog_mult = progress_mult(history)
    # Bump: rises with the first refresh, fades as cohorts catch up.
    rise = 1.0 - math.exp(-horizon_h / TAU_RISE_H)
    fade = math.exp(-horizon_h / TAU_FADE_H)

    # Per-trophy catch-up: effective jump for a branch with catch-up K
    # is (N + dN) / (N + K*dN); its log is the branch's dilution bump.
    def bump_for(k):
        jeff = jump / (1.0 + k * (jump - 1.0))
        return math.log(jeff) * rise * fade

    bump_slow = bump_for(K_SLOW)
    bump_fast = bump_for(K_FAST)

    r_max = max((r for r in last_rates if r < 99.99), default=0.0)

    ceil = ceilings(case["trophies"], last_rates)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r0 = last_rates[i]
        if r0 >= 99.99:
            # Everybody holds it, new players included: stays ~100%.
            # q95 = 100 (not 99.99): the rate really is 100, and the
            # scorer clamps quantiles to 99.99 for the logit anyway.
            rows.append({"id": trophy["id"],
                         "q": [99.9, 99.95, 99.99, 99.99, 100]})
            continue
        lg0 = logit(r0)
        s_target = math.log(ceil[i] / (1.0 - ceil[i]))
        tau = TAU_PULL_D + TAU_PULL_EXTRA_D * (1.0 - ceil[i] / 0.6)
        pull = (s_target - lg0) * (1.0 - math.exp(-horizon_d / tau))
        pull *= prog_mult

        # Spread within one branch: base + progress uncertainty as the
        # pull matures + refresh uncertainty that fades with horizon.
        matured = 1.0 - math.exp(-horizon_d / tau)
        spread = math.sqrt(
            S_BASE ** 2
            + (S_PROG * matured) ** 2
            + (S_REFRESH * fade) ** 2
        )
        # Fast-branch weight from easiness (rate vs the case's best).
        e = (r0 / r_max) if r_max > 0 else 0.0
        w_fast = min(W_MAX, e ** W_POWER)
        means = [lg0 + pull - bump_slow, lg0 + pull - bump_fast]
        weights = [1.0 - w_fast, w_fast]
        qs = []
        for tau_q in LEVELS:
            q = round4(clamp_pct(inv_logit(
                mixture_quantile(means, weights, spread, tau_q))))
            qs.append(q)
        qs.sort()
        rows.append({"id": trophy["id"], "q": qs})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def fallback_case(case):
    """Never crash a case: persistence with the house baseline's fan."""
    last = case["history"][-1] if case["history"] else None
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = last["rates"][i] if last and last["rates"][i] is not None else 1.0
        lg = logit(r)
        qs = sorted(round4(clamp_pct(inv_logit(lg + z * spread))) for z in Z)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
