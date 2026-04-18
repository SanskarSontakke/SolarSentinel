"""
SolarSentinel — Agent Benchmark Tool
Answers: "Is version B better than version A?"

Usage:
    python benchmark.py --agent-a submission.py --agent-b snapshots/agent_v3.py --games 40 --mode 2p
    python benchmark.py --agent-a submission.py --agent-b snapshots/agent_v3.py --quick
    python benchmark.py --agent-a submission.py --agent-b snapshots/agent_v3.py --games 20 --mode 4p
"""

import os
import sys
import json
import math
import time
import logging
import warnings
import argparse

# Force silence all INFO logs from kaggle_environments and absl
logging.getLogger("kaggle_environments").setLevel(logging.WARNING)
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass
logging.disable(logging.INFO)
warnings.filterwarnings("ignore")

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_agent(filepath: str):
    """Dynamically import an agent function from a .py file."""
    path = Path(filepath)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        print(f"[ERROR] Agent file not found: {path}")
        sys.exit(1)
    module_name = f"agent_{path.stem}_{id(path)}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        print(f"[ERROR] No 'agent' function found in {path}")
        sys.exit(1)
    return mod.agent


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """
    Wilson score interval for a binomial proportion at 95% confidence.
    Returns (center, lower, upper).
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    p_hat = wins / total
    denom = 1 + z * z / total
    center = (p_hat + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total) / denom
    return center, max(0.0, center - spread), min(1.0, center + spread)


def chi_square_test(wins_a: int, wins_b: int, draws: int) -> tuple[float, float, bool]:
    """
    Chi-square goodness-of-fit test on win counts.
    H0: both agents win equally often.
    Returns (chi2_stat, p_value, is_significant).
    """
    # Try scipy first
    try:
        from scipy.stats import chisquare
        total_decisive = wins_a + wins_b
        if total_decisive == 0:
            return 0.0, 1.0, False
        expected = total_decisive / 2.0
        stat, p = chisquare([wins_a, wins_b], f_exp=[expected, expected])
        return float(stat), float(p), p < 0.05
    except ImportError:
        pass

    # Manual chi-square (1 degree of freedom)
    total_decisive = wins_a + wins_b
    if total_decisive == 0:
        return 0.0, 1.0, False
    expected = total_decisive / 2.0
    chi2 = ((wins_a - expected) ** 2 + (wins_b - expected) ** 2) / expected

    # Approximate p-value from chi2 with 1 df using survival function
    # P(X > chi2) ≈ erfc(sqrt(chi2/2)) for 1 df
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return chi2, p_value, p_value < 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# GAME RUNNER — 2-PLAYER
# ═══════════════════════════════════════════════════════════════════════════════

def run_2p_game(agent_a, agent_b, a_is_p0: bool) -> dict:
    """
    Run a single 2-player orbit_wars game.
    Returns result dict from A's perspective.
    """
    from kaggle_environments import make

    env = make("orbit_wars", debug=False)

    if a_is_p0:
        env.run([agent_a, agent_b])
        a_idx, b_idx = 0, 1
    else:
        env.run([agent_b, agent_a])
        a_idx, b_idx = 1, 0

    final = env.steps[-1]
    game_length = len(env.steps) - 1

    r_a = final[a_idx].reward or 0
    r_b = final[b_idx].reward or 0

    obs = final[0].observation
    planets = obs.planets if hasattr(obs, "planets") else obs.get("planets", [])

    ships_a = sum(p[5] for p in planets if p[1] == a_idx)
    ships_b = sum(p[5] for p in planets if p[1] == b_idx)

    if r_a > r_b:
        outcome = "A"
    elif r_b > r_a:
        outcome = "B"
    else:
        outcome = "DRAW"

    return {
        "outcome": outcome,
        "ships_a": ships_a,
        "ships_b": ships_b,
        "ship_lead": ships_a - ships_b,
        "game_length": game_length,
        "a_slot": 0 if a_is_p0 else 1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GAME RUNNER — 4-PLAYER (FFA)
# ═══════════════════════════════════════════════════════════════════════════════

def run_4p_game(agent_a, agent_b, game_num: int) -> dict:
    """
    Run a single 4-player orbit_wars game with 2 copies of A and 2 copies of B.
    Slot assignment rotates based on game_num.
    Returns result from A-team's perspective.
    """
    from kaggle_environments import make

    env = make("orbit_wars", debug=False, configuration={"episodeSteps": 500})

    # Alternate slot arrangements
    if game_num % 2 == 0:
        agents = [agent_a, agent_b, agent_a, agent_b]
        a_slots = {0, 2}
        b_slots = {1, 3}
    else:
        agents = [agent_b, agent_a, agent_b, agent_a]
        a_slots = {1, 3}
        b_slots = {0, 2}

    env.run(agents)

    final = env.steps[-1]
    game_length = len(env.steps) - 1

    obs = final[0].observation
    planets = obs.planets if hasattr(obs, "planets") else obs.get("planets", [])

    ships_a = sum(p[5] for p in planets if p[1] in a_slots)
    ships_b = sum(p[5] for p in planets if p[1] in b_slots)

    # Team with more total ships wins
    if ships_a > ships_b:
        outcome = "A"
    elif ships_b > ships_a:
        outcome = "B"
    else:
        outcome = "DRAW"

    return {
        "outcome": outcome,
        "ships_a": ships_a,
        "ships_b": ships_b,
        "ship_lead": ships_a - ships_b,
        "game_length": game_length,
        "a_slots": sorted(a_slots),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(args):
    agent_a = load_agent(args.agent_a)
    agent_b = load_agent(args.agent_b)

    num_games = 10 if args.quick else args.games
    skip_stats = args.quick

    print("=" * 72)
    print("  SolarSentinel — Agent Benchmark")
    print("=" * 72)
    print(f"  Agent A:  {args.agent_a}")
    print(f"  Agent B:  {args.agent_b}")
    print(f"  Mode:     {args.mode}")
    print(f"  Games:    {num_games}{'  (quick mode — stats skipped)' if skip_stats else ''}")
    print("=" * 72)

    results = []
    wins_a, wins_b, draws = 0, 0, 0
    t0 = time.time()

    for i in range(1, num_games + 1):
        if args.mode == "2p":
            a_is_p0 = (i % 2 == 1)  # alternate sides
            result = run_2p_game(agent_a, agent_b, a_is_p0)
        else:
            result = run_4p_game(agent_a, agent_b, i)

        results.append(result)

        if result["outcome"] == "A":
            wins_a += 1
        elif result["outcome"] == "B":
            wins_b += 1
        else:
            draws += 1

        # Progress every 5 games or on last game
        if i % 5 == 0 or i == num_games or i == 1:
            elapsed = time.time() - t0
            eta = (elapsed / i) * (num_games - i) if i > 0 else 0
            print(
                f"  [{i:3d}/{num_games}]  A:{wins_a} B:{wins_b} D:{draws}  |  "
                f"Ships: {result['ships_a']:.0f} vs {result['ships_b']:.0f}  |  "
                f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left"
            )

    elapsed = time.time() - t0
    total = wins_a + wins_b + draws
    decisive = wins_a + wins_b

    # ── Compute metrics ───────────────────────────────────────────────────
    wr_a_center, wr_a_lo, wr_a_hi = wilson_interval(wins_a, total)
    wr_b_center, wr_b_lo, wr_b_hi = wilson_interval(wins_b, total)

    avg_ship_lead = sum(r["ship_lead"] for r in results) / max(1, len(results))
    avg_game_len = sum(r["game_length"] for r in results) / max(1, len(results))

    if skip_stats:
        chi2, p_val, significant = 0.0, 1.0, False
        verdict = "QUICK MODE — no statistical verdict"
    else:
        chi2, p_val, significant = chi_square_test(wins_a, wins_b, draws)
        if significant:
            verdict = "A is BETTER" if wins_a > wins_b else "B is BETTER"
        else:
            verdict = "NO SIGNIFICANT DIFFERENCE"

    # ── Print report ──────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  BENCHMARK RESULTS")
    print("=" * 72)
    print(f"  Agent A:  {args.agent_a}")
    print(f"  Agent B:  {args.agent_b}")
    print(f"  Games:    {total}  ({args.mode})")
    print(f"  Time:     {elapsed:.1f}s  ({elapsed / total:.1f}s per game)")
    print("-" * 72)
    print(f"  Wins A:   {wins_a:4d}  ({wins_a/total*100:5.1f}%)  "
          f"95% CI: [{wr_a_lo*100:4.1f}%, {wr_a_hi*100:4.1f}%]")
    print(f"  Wins B:   {wins_b:4d}  ({wins_b/total*100:5.1f}%)  "
          f"95% CI: [{wr_b_lo*100:4.1f}%, {wr_b_hi*100:4.1f}%]")
    print(f"  Draws:    {draws:4d}  ({draws/total*100:5.1f}%)")
    print("-" * 72)
    print(f"  Avg ship lead (A − B):  {avg_ship_lead:+.0f}")
    print(f"  Avg game length:        {avg_game_len:.0f} steps")
    if not skip_stats:
        print("-" * 72)
        print(f"  Chi-square statistic:   {chi2:.3f}")
        print(f"  p-value:                {p_val:.4f}")
        print(f"  Significant (p<0.05):   {'YES' if significant else 'NO'}")
    print("-" * 72)
    print(f"  ╔═══════════════════════════════════════════╗")
    print(f"  ║  VERDICT:  {verdict:<33s}║")
    print(f"  ╚═══════════════════════════════════════════╝")
    print("=" * 72)

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_a": str(args.agent_a),
        "agent_b": str(args.agent_b),
        "mode": args.mode,
        "total_games": total,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "winrate_a": round(wr_a_center, 4),
        "winrate_a_ci": [round(wr_a_lo, 4), round(wr_a_hi, 4)],
        "winrate_b": round(wr_b_center, 4),
        "winrate_b_ci": [round(wr_b_lo, 4), round(wr_b_hi, 4)],
        "avg_ship_lead": round(avg_ship_lead, 1),
        "avg_game_length": round(avg_game_len, 1),
        "chi2": round(chi2, 4),
        "p_value": round(p_val, 4),
        "significant": significant,
        "verdict": verdict,
        "elapsed_seconds": round(elapsed, 1),
        "per_game_results": results,
    }

    out_path = ROOT / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SolarSentinel — Agent Benchmark: Is B better than A?"
    )
    parser.add_argument(
        "--agent-a", required=True, dest="agent_a",
        help="Path to agent A's .py file (e.g. submission.py)"
    )
    parser.add_argument(
        "--agent-b", required=True, dest="agent_b",
        help="Path to agent B's .py file (e.g. snapshots/agent_v3.py)"
    )
    parser.add_argument(
        "--games", type=int, default=40,
        help="Number of games to run (default: 40)"
    )
    parser.add_argument(
        "--mode", choices=["2p", "4p"], default="2p",
        help="2-player head-to-head or 4-player FFA (default: 2p)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick sanity check: 10 games, no statistical tests"
    )
    args = parser.parse_args()

    run_benchmark(args)
