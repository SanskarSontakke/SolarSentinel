"""
SolarSentinel — Self-Play Arena + Parameter Evolution

Modes:
    arena   — ELO-rated self-play measurement
    evolve  — CMA-ES-style parameter evolution

Usage:
    python self_play_trainer.py --mode arena  --games 500 --snapshot-every 50
    python self_play_trainer.py --mode evolve --generations 30 --pop-size 8 --games-per-eval 20
"""

import os
import re
import sys
import json
import math
import time
import shutil
import random
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

import textwrap
import tempfile
import subprocess
import importlib
import importlib.util
import multiprocessing
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SNAPSHOTS_DIR = ROOT / "snapshots"
ELO_FILE = ROOT / "elo_history.json"
EVOLUTION_FILE = ROOT / "evolution_history.json"
SUBMISSION_FILE = ROOT / "submission.py"
TEMP_DIR = ROOT / ".evolution_tmp"


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def load_agent_from_file(filepath):
    """Dynamically import an agent function from a .py file."""
    path = Path(filepath)
    module_name = f"agent_{path.stem}_{id(path)}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def save_snapshot(version: int) -> Path:
    ensure_dir(SNAPSHOTS_DIR)
    dest = SNAPSHOTS_DIR / f"agent_v{version}.py"
    shutil.copy2(SUBMISSION_FILE, dest)
    print(f"  [SNAPSHOT] Saved agent_v{version}.py")
    return dest


# ═══════════════════════════════════════════════════════════════════════════════
#  MODE 1: ARENA (ELO-rated self-play)
# ═══════════════════════════════════════════════════════════════════════════════

class OpponentPool:
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.versions: list[int] = []
        self.agents_paths: dict[int, Path] = {}

    def add(self, version: int, path: Path):
        self.versions.append(version)
        self.agents_paths[version] = path
        while len(self.versions) > self.max_size:
            old = self.versions.pop(0)
            del self.agents_paths[old]

    def pick(self) -> tuple[int, Path]:
        if not self.versions:
            # Fallback to agent_v0 if pool is empty
            return 0, SNAPSHOTS_DIR / "agent_v0.py"
        ver = random.choice(self.versions)
        return ver, self.agents_paths[ver]


class ELOTracker:
    def __init__(self, k: int = 32):
        self.k = k
        self.ratings: dict[str, float] = {}
        self.history: list[dict] = []

    def _ensure(self, name):
        if name not in self.ratings:
            self.ratings[name] = 1000.0

    def update(self, a: str, b: str, result: int):
        self._ensure(a); self._ensure(b)
        ra, rb = self.ratings[a], self.ratings[b]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        sa = {0: 1.0, 1: 0.0, -1: 0.5}[result]
        self.ratings[a] = ra + self.k * (sa - ea)
        self.ratings[b] = rb + self.k * ((1 - sa) - (1 - ea))
        self.history.append({
            "game": len(self.history) + 1, "a": a, "b": b,
            "result": result, "ra": round(self.ratings[a], 1),
            "rb": round(self.ratings[b], 1),
        })

    def get(self, name): self._ensure(name); return self.ratings[name]

    def save(self):
        with open(ELO_FILE, "w") as f:
            json.dump({"ratings": self.ratings, "history": self.history}, f, indent=2)

    def load(self):
        if ELO_FILE.exists():
            d = json.loads(ELO_FILE.read_text())
            self.ratings = {k: float(v) for k, v in d.get("ratings", {}).items()}
            self.history = d.get("history", [])


# ── Batch game evaluation ─────────────────────────────────────────────────────

def _eval_worker(job):
    """
    Worker function for multiprocessing.
    job = (agent_a_path, agent_b_path, game_idx, meta)
    Returns: (winner, ship_lead, meta)
    """
    path_a, path_b, game_idx, meta = job
    try:
        from kaggle_environments import make

        # Load agents in the subprocess
        agent_a = load_agent_from_file(path_a)
        agent_b = load_agent_from_file(path_b)

        env = make("orbit_wars", debug=False)
        swap = (game_idx % 2 == 0)
        if swap:
            env.run([agent_b, agent_a])
            ai, bi = 1, 0
        else:
            env.run([agent_a, agent_b])
            ai, bi = 0, 1

        final = env.steps[-1]
        ra = final[ai].reward or 0
        rb = final[bi].reward or 0
        obs = final[0].observation
        planets = obs.planets if hasattr(obs, "planets") else obs.get("planets", [])
        sa = sum(p[5] for p in planets if p[1] == ai)
        sb = sum(p[5] for p in planets if p[1] == bi)

        # Result from perspective of agent_a
        won = 1 if ra > rb else (0 if rb > ra else 0.5)
        return won, sa - sb, meta
    except Exception as e:
        return 0, 0, meta


def run_arena(args):
    print("=" * 70)
    print("  SolarSentinel — Self-Play Arena (Parallel Optimized)")
    print("=" * 70)
    print(f"  Max games:       {args.games}")
    print(f"  Snapshot every:  {args.snapshot_every}")
    print(f"  Pool size:       {args.pool_size}")
    print(f"  Batch size:      24")
    print("=" * 70)

    elo = ELOTracker(k=args.elo_k); elo.load()
    pool = OpponentPool(max_size=args.pool_size)

    # Initial setup
    v0_path = SNAPSHOTS_DIR / "agent_v0.py"
    if not v0_path.exists():
        save_snapshot(0)
    pool.add(0, v0_path)
    
    ver = 0
    wins, losses, draws = 0, 0, 0
    recent_elos = deque(maxlen=args.stagnation)
    t0 = time.time()
    n_workers = multiprocessing.cpu_count()
    
    print(f"\n  Starting arena at ELO {elo.get('current'):.0f}\n" + "-" * 70)

    # Run in batches of 24 games to keep it parallel
    batch_size = 24
    for batch_start in range(1, args.games + 1, batch_size):
        jobs = []
        batch_end = min(batch_start + batch_size - 1, args.games)
        
        for g in range(batch_start, batch_end + 1):
            ov, opp_path = pool.pick()
            jobs.append((str(SUBMISSION_FILE), str(opp_path), g, ov))

        with multiprocessing.Pool(n_workers) as p:
            results = p.map(_eval_worker, jobs)

        for won, lead, ov in results:
            # result: 0 if A wins, 1 if B wins, -1 if draw
            # won: 1 if A, 0 if B, 0.5 if draw
            res_val = 0 if won == 1 else (1 if won == 0 else -1)
            elo.update("current", f"v{ov}", res_val)
            
            if won == 1: wins += 1
            elif won == 0: losses += 1
            else: draws += 1
            
            ce = elo.get("current"); recent_elos.append(ce)
            
        elo.save()
        last_ov = results[-1][2]
        last_won = results[-1][0]
        tag = "WIN " if last_won == 1 else ("LOSS" if last_won == 0 else "DRAW")
        
        print(f"  Batch {batch_start:4d}-{batch_end:4d} | ELO: {ce:7.0f} | W/L/D: {wins}/{losses}/{draws} | "
              f"{time.time()-t0:.0f}s")

        # Snapshot logic - check if we passed a threshold
        if (batch_end // args.snapshot_every) > (batch_start // args.snapshot_every):
            ver += 1
            v_path = save_snapshot(ver)
            pool.add(ver, v_path)
            elo.ratings[f"v{ver}"] = ce
            print(f"  [POOL] +v{ver} (ELO {ce:.0f})")

        if len(recent_elos) >= args.stagnation:
            if max(recent_elos) - min(recent_elos) < 5.0:
                print(f"\n  [STOP] ELO stagnated."); break

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  ARENA COMPLETE  |  ELO: {elo.get('current'):.0f}  |  {elapsed:.0f}s")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODE 2: PARAMETER EVOLUTION (CMA-ES style)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Parameter space with bounds ───────────────────────────────────────────────
PARAM_SPACE = {
    "enemy_multiplier":              (1.2, 3.5),
    "finishing_multiplier":          (1.0, 3.0),
    "early_neutral_multiplier":      (1.0, 3.0),
    "safe_neutral_early_multiplier": (0.8, 2.5),
    "contested_neutral_penalty":     (0.05, 0.8),
    "prod_weight":                   (5.0, 30.0),
    "iw_weight":                     (0.5, 8.0),
    "contested_margin":              (1.0, 2.5),
    "cost_turns_weight":             (0.1, 2.0),
    "funnel_finishing_ratio":        (0.5, 0.95),
    "funnel_ratio":                  (0.3, 0.85),
}

PARAM_NAMES = list(PARAM_SPACE.keys())

# ── Current defaults (read from submission.py at startup) ─────────────────────
DEFAULT_CFG = {
    "enemy_multiplier": 2.0,
    "finishing_multiplier": 1.5,
    "early_neutral_multiplier": 1.6,
    "safe_neutral_early_multiplier": 1.4,
    "contested_neutral_penalty": 0.25,
    "prod_weight": 15.0,
    "iw_weight": 3.0,
    "contested_margin": 1.4,
    "cost_turns_weight": 0.5,
    "funnel_finishing_ratio": 0.80,
    "funnel_ratio": 0.65,
}


def clamp_cfg(cfg: dict) -> dict:
    """Clamp every parameter to its defined bounds."""
    out = {}
    for k in PARAM_NAMES:
        lo, hi = PARAM_SPACE[k]
        out[k] = float(np.clip(cfg.get(k, DEFAULT_CFG[k]), lo, hi))
    return out


def cfg_to_vec(cfg: dict) -> np.ndarray:
    return np.array([cfg[k] for k in PARAM_NAMES], dtype=np.float64)


def vec_to_cfg(vec) -> dict:
    if isinstance(vec, dict):
        return clamp_cfg(vec)
    cfg = {k: float(vec[i]) for i, k in enumerate(PARAM_NAMES)}
    return clamp_cfg(cfg)


def mutate_around(parent_vec: np.ndarray, sigma: float, rng: np.random.Generator) -> dict:
    """Sample one child around a parent vector with Gaussian noise."""
    noise = rng.normal(0, 1, size=len(parent_vec))
    # Scale noise per-dimension by the parameter range
    scales = np.array([PARAM_SPACE[k][1] - PARAM_SPACE[k][0] for k in PARAM_NAMES])
    child_vec = parent_vec + sigma * noise * scales
    return vec_to_cfg(dict(zip(PARAM_NAMES, child_vec)))


# ── Temporary agent file creation ────────────────────────────────────────────

def make_temp_agent(cfg: dict, idx: int) -> Path:
    """
    Create a temporary copy of submission.py with hard-coded CFG values.
    The trick: we inject the config as an override_config default.
    """
    ensure_dir(TEMP_DIR)
    src = SUBMISSION_FILE.read_text()

    # Find and replace the CFG dict literal inside agent()
    # We replace the block between `CFG = {` ... `}` with our values
    cfg_str = "    CFG = {\n"
    for k in PARAM_NAMES:
        cfg_str += f'        "{k}": {cfg[k]:.6f},\n'
    cfg_str += "    }"

    # Use regex to replace the CFG dict block
    pattern = r'    CFG = \{[^}]+\}'
    new_src = re.sub(pattern, cfg_str, src, count=1)

    dest = TEMP_DIR / f"candidate_{idx}.py"
    dest.write_text(new_src)
    return dest


# ── Parallel game evaluation ─────────────────────────────────────────────────

# (evaluate_candidate is removed in favor of generation-wide batching)


# ── Write best CFG back into submission.py ───────────────────────────────────

def inject_cfg_into_submission(cfg: dict):
    """Replace the CFG dict inside submission.py with the evolved values."""
    src = SUBMISSION_FILE.read_text()

    cfg_str = "    CFG = {\n"
    for k in PARAM_NAMES:
        # Use clean formatting: align values
        cfg_str += f'        "{k}": {cfg[k]:.4f},\n'
    cfg_str += "    }"

    pattern = r'    CFG = \{[^}]+\}'
    new_src = re.sub(pattern, cfg_str, src, count=1)

    SUBMISSION_FILE.write_text(new_src)
    print(f"  [INJECTED] Updated CFG in submission.py")


# ── Main evolution loop ──────────────────────────────────────────────────────

def run_evolution(args):
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("  SolarSentinel — Parameter Evolution (CMA-ES style)")
    print("=" * 70)
    print(f"  Generations:     {args.generations}")
    print(f"  Population:      {args.pop_size}")
    print(f"  Games/eval:      {args.games_per_eval}")
    print(f"  Initial sigma:   {args.sigma}")
    print(f"  Sigma decay:     {args.sigma_decay}")
    print(f"  Workers:         {multiprocessing.cpu_count()}")
    print("=" * 70)

    # Save baseline snapshot as opponent
    ensure_dir(SNAPSHOTS_DIR)
    opp_path = SNAPSHOTS_DIR / "agent_v0.py"
    if not opp_path.exists():
        save_snapshot(0)

    # Initialize from current defaults
    best_cfg = deepcopy(DEFAULT_CFG)
    best_score = -1e9
    best_vec = cfg_to_vec(best_cfg)
    sigma = args.sigma

    # Parents: start with 2 copies of the default
    parent_vecs = [cfg_to_vec(DEFAULT_CFG), cfg_to_vec(DEFAULT_CFG)]

    history = []
    stagnation_counter = 0
    n_workers = max(1, multiprocessing.cpu_count() - 1)
    t0 = time.time()

    for gen in range(1, args.generations + 1):
        gen_t0 = time.time()
        print(f"\n{'─'*70}")
        print(f"  Generation {gen}/{args.generations}  |  σ = {sigma:.4f}  |  "
              f"Best score: {best_score:.1f}")
        print(f"{'─'*70}")

        # ── Generate population ───────────────────────────────────────────
        population = []

        # Keep parents unchanged (elitism)
        for pi, pv in enumerate(parent_vecs):
            population.append(vec_to_cfg(pv))

        # Fill rest with mutations from random parents
        while len(population) < args.pop_size:
            parent = parent_vecs[rng.integers(0, len(parent_vecs))]
            child = mutate_around(parent, sigma, rng)
            population.append(child)

        # ── Evaluate population in one massive batch ──────────────────────
        all_jobs = []
        cand_paths = []
        
        # Determine opponent: pick a random snapshot from the history
        snapshot_files = sorted(list(SNAPSHOTS_DIR.glob("agent_v*.py")), key=lambda x: x.name)
        
        for i, cfg in enumerate(population):
            cand_path = make_temp_agent(cfg, i)
            cand_paths.append(cand_path)
            
            # Use random snapshot for evaluation (ensure agent_v0 exists)
            opp_choice = random.choice(snapshot_files) if snapshot_files else snapshot_files[0]
            
            for g in range(args.games_per_eval):
                all_jobs.append((str(cand_path), str(opp_choice), g, i))

        print(f"    Running {len(all_jobs)} simulation games in parallel...")
        with multiprocessing.Pool(n_workers) as pool:
            all_results = pool.map(_eval_worker, all_jobs)

        # Re-group results
        results_by_cand = [[] for _ in range(len(population))]
        for won, lead, cand_idx in all_results:
            results_by_cand[cand_idx].append((won, lead))

        scores = []
        for i, res_list in enumerate(results_by_cand):
            wr = sum(r[0] for r in res_list) / len(res_list)
            avg_lead = sum(r[1] for r in res_list) / len(res_list)
            score = wr * 100 + avg_lead / 50.0
            scores.append(score)

            tag = "★" if i < len(parent_vecs) else " "
            print(f"    {tag} [{i+1:2d}/{args.pop_size}]  "
                  f"WR: {wr*100:5.1f}%  Lead: {avg_lead:+7.0f}  "
                  f"Score: {score:7.1f}")

        # ── Selection: top 2 become parents ───────────────────────────────
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        parent_vecs = [cfg_to_vec(population[ranked[0]]),
                       cfg_to_vec(population[ranked[1]])]

        gen_best_score = scores[ranked[0]]
        gen_best_cfg = population[ranked[0]]

        # ── Track improvement ─────────────────────────────────────────────
        improved = gen_best_score > best_score + 0.1
        if improved:
            best_score = gen_best_score
            best_cfg = deepcopy(gen_best_cfg)
            best_vec = cfg_to_vec(best_cfg)
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Decay sigma
        sigma *= args.sigma_decay

        gen_elapsed = time.time() - gen_t0
        print(f"\n  Gen {gen} best: score={gen_best_score:.1f}  "
              f"{'↑ NEW BEST' if improved else '(no improvement)'}"
              f"  |  stag={stagnation_counter}  |  {gen_elapsed:.0f}s")

        # ── Record history ────────────────────────────────────────────────
        history.append({
            "generation": gen,
            "best_score": round(gen_best_score, 2),
            "global_best_score": round(best_score, 2),
            "sigma": round(sigma, 6),
            "stagnation": stagnation_counter,
            "best_cfg": {k: round(v, 6) for k, v in gen_best_cfg.items()},
            "all_scores": [round(s, 2) for s in scores],
            "elapsed_s": round(gen_elapsed, 1),
        })

        # Save history incrementally
        with open(EVOLUTION_FILE, "w") as f:
            json.dump({"best_cfg": best_cfg, "best_score": best_score,
                        "history": history}, f, indent=2)

        # ── Early stop ────────────────────────────────────────────────────
        if stagnation_counter >= 5:
            print(f"\n  [STOP] No improvement in 5 generations. Stopping.")
            break

    # ── Cleanup temp files ────────────────────────────────────────────────
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    # ── Print final results ───────────────────────────────────────────────
    total_elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  EVOLUTION COMPLETE")
    print("=" * 70)
    print(f"  Generations run:   {len(history)}")
    print(f"  Best score:        {best_score:.1f}")
    print(f"  Total time:        {total_elapsed:.0f}s")
    print(f"\n  Best CFG found:")
    for k, v in best_cfg.items():
        lo, hi = PARAM_SPACE[k]
        bar_pos = (v - lo) / (hi - lo)
        bar = "▓" * int(bar_pos * 20) + "░" * (20 - int(bar_pos * 20))
        print(f"    {k:>35s}  {v:8.4f}  [{lo:5.2f} {bar} {hi:5.2f}]")

    # ── Inject into submission.py ─────────────────────────────────────────
    print(f"\n  Injecting best CFG into submission.py...")
    inject_cfg_into_submission(best_cfg)

    # ── Verification benchmark ────────────────────────────────────────────
    print(f"\n  Running verification benchmark...")
    benchmark_cmd = [
        sys.executable, str(ROOT / "benchmark.py"),
        "--agent-a", str(SUBMISSION_FILE),
        "--agent-b", str(SNAPSHOTS_DIR / "agent_v0.py"),
        "--quick",
    ]
    try:
        result = subprocess.run(benchmark_cmd, cwd=str(ROOT),
                                capture_output=False, text=True, timeout=300)
    except Exception as e:
        print(f"  [WARN] Benchmark failed: {e}")

    print("\n  Evolution history saved to evolution_history.json")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SolarSentinel — Self-Play Arena + Parameter Evolution"
    )
    sub = parser.add_subparsers(dest="mode", help="Operating mode")

    # ── Arena mode ────────────────────────────────────────────────────────
    arena_p = sub.add_parser("arena", help="ELO-rated self-play measurement")
    arena_p.add_argument("--games", type=int, default=500)
    arena_p.add_argument("--snapshot-every", type=int, default=50, dest="snapshot_every")
    arena_p.add_argument("--pool-size", type=int, default=5, dest="pool_size")
    arena_p.add_argument("--stagnation", type=int, default=100)
    arena_p.add_argument("--elo-k", type=int, default=32, dest="elo_k")

    # ── Evolve mode ───────────────────────────────────────────────────────
    evo_p = sub.add_parser("evolve", help="CMA-ES-style parameter evolution")
    evo_p.add_argument("--generations", type=int, default=30)
    evo_p.add_argument("--pop-size", type=int, default=8, dest="pop_size")
    evo_p.add_argument("--games-per-eval", type=int, default=20, dest="games_per_eval")
    evo_p.add_argument("--sigma", type=float, default=0.3,
                       help="Initial mutation step size (default: 0.3)")
    evo_p.add_argument("--sigma-decay", type=float, default=0.95, dest="sigma_decay",
                       help="Sigma multiplier per generation (default: 0.95)")

    # ── Legacy flat flags (backwards compat) ──────────────────────────────
    parser.add_argument("--mode", dest="mode_flat", default=None,
                        choices=["arena", "evolve"],
                        help="Alternative to subcommands")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--snapshot-every", type=int, default=50, dest="snapshot_every")
    parser.add_argument("--pool-size", type=int, default=5, dest="pool_size")
    parser.add_argument("--stagnation", type=int, default=100)
    parser.add_argument("--elo-k", type=int, default=32, dest="elo_k")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--pop-size", type=int, default=8, dest="pop_size")
    parser.add_argument("--games-per-eval", type=int, default=20, dest="games_per_eval")
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--sigma-decay", type=float, default=0.95, dest="sigma_decay")

    args = parser.parse_args()

    # Resolve mode
    mode = args.mode or args.mode_flat
    if mode is None:
        parser.print_help()
        print("\n  Error: specify a mode.  E.g.:")
        print("    python self_play_trainer.py arena --games 100")
        print("    python self_play_trainer.py evolve --generations 10")
        sys.exit(1)

    if mode == "arena":
        run_arena(args)
    elif mode == "evolve":
        run_evolution(args)
