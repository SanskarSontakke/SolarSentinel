"""
SolarSentinel — Parallel Agent Benchmark Tool
Supports 2p, 4p FFA, and N-player multi-agent tests with 12-thread parallelism.
"""

import os
import sys
import json
import math
import time
import logging
import warnings
import argparse
import multiprocessing
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import importlib.util

# Force silence
logging.disable(logging.INFO)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent

def load_agent(filepath: str):
    path = Path(filepath)
    if not path.is_absolute(): path = ROOT / path
    if not path.exists():
        print(f"[ERROR] Agent file not found: {path}"); sys.exit(1)
    module_name = f"agent_{path.stem}_{id(path)}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent

def wilson_interval(wins: int, total: int):
    if total == 0: return 0.0, 0.0, 0.0
    p = wins / total; z = 1.96; d = 1 + z*z/total
    c = (p + z*z/(2*total))/d
    s = z * math.sqrt((p*(1-p) + z*z/(4*total))/total)/d
    return c, max(0.0, c-s), min(1.0, c+s)

def run_single_game(args_tuple):
    """Worker function for parallel execution."""
    mode, agents_paths, game_idx, seed = args_tuple
    from kaggle_environments import make
    
    # Reload agents in child process
    agents = [load_agent(p) for p in agents_paths]
    
    env = make("orbit_wars", debug=False)
    # Set seed for reproducibility in sets
    if seed is not None:
        # Note: orbit_wars might not respect a global seed easily, but we try
        pass
        
    env.run(agents)
    final = env.steps[-1]
    
    # Rewards and ships
    rewards = [s.reward or 0 for s in final]
    obs = final[0].observation
    planets = obs.planets if hasattr(obs, "planets") else obs.get("planets", [])
    
    ship_counts = [0] * len(agents)
    for p in planets:
        if p[1] != -1: ship_counts[p[1]] += p[5]
    
    # Determine winner (highest reward, then highest ships)
    max_r = max(rewards)
    winners = [i for i, r in enumerate(rewards) if r == max_r]
    if len(winners) > 1:
        max_s = max(ship_counts[i] for i in winners)
        winners = [i for i in winners if ship_counts[i] == max_s]
    
    winner_idx = winners[0] if winners else -1
    
    return {
        "game_idx": game_idx,
        "rewards": rewards,
        "ships": ship_counts,
        "winner": winner_idx,
        "steps": len(env.steps) - 1
    }

def run_benchmark(args):
    t0 = time.time()
    
    # Define agent list for the mode
    agent_paths = []
    if args.mode == "2p":
        agent_paths = [args.agent_a, args.agent_b]
    elif args.mode == "4p_team":
        # team mode: 2 of A, 2 of B (alternating)
        agent_paths = [args.agent_a, args.agent_b, args.agent_a, args.agent_b]
    elif args.mode == "1v3":
        # 1 vs 3 of B
        agent_paths = [args.agent_a, args.agent_b, args.agent_b, args.agent_b]
    elif args.mode == "ffa":
        # Custom agent list
        agent_paths = args.agents_list if args.agents_list else [args.agent_a, args.agent_b]
    
    num_games = 10 if args.quick else args.games
    print("=" * 72)
    print(f"  SolarSentinel Parallel Benchmark (Threads: {args.threads})")
    print("=" * 72)
    print(f"  Mode:   {args.mode}")
    print(f"  Agents: {[Path(p).name for p in agent_paths]}")
    print(f"  Games:  {num_games}")
    print("=" * 72)

    # Prepare jobs
    jobs = []
    for i in range(num_games):
        # Rotate slots if 2p or 4p team
        current_agents = list(agent_paths)
        if args.mode == "2p" and i % 2 == 1:
            current_agents = [agent_paths[1], agent_paths[0]]
        elif args.mode == "4p_team" and i % 2 == 1:
            current_agents = [agent_paths[1], agent_paths[0], agent_paths[1], agent_paths[0]]
        
        jobs.append((args.mode, current_agents, i, i))

    # Run parallel
    with multiprocessing.Pool(args.threads) as pool:
        results = pool.map(run_single_game, jobs)

    # Process results
    wins_per_agent = [0] * len(agent_paths)
    team_a_wins, team_b_wins = 0, 0
    
    for i, res in enumerate(results):
        w = res["winner"]
        # Map winner back to original agent perspective
        if args.mode == "2p":
            orig_winner = w if i % 2 == 0 else (1 - w)
            wins_per_agent[orig_winner] += 1
        elif args.mode == "4p_team":
            # 0, 2 are Team A; 1, 3 are Team B (unswapped)
            if i % 2 == 0:
                if w in (0, 2): team_a_wins += 1
                else: team_b_wins += 1
            else:
                # 0, 2 are Team B; 1, 3 are Team A (swapped)
                if w in (1, 3): team_a_wins += 1
                else: team_b_wins += 1
        elif args.mode == "1v3":
            wins_per_agent[w] += 1
        else:
            wins_per_agent[w] += 1

    elapsed = time.time() - t0
    
    # FFA/1v3 reporting
    print(f"\nFinal Win Breakdown (Time: {elapsed:.1f}s):")
    for i, wins in enumerate(wins_per_agent):
        p_name = Path(agent_paths[i]).name
        print(f"  P{i} ({p_name}): {wins} wins ({wins/num_games*100:.1f}%)")
    
    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "games": num_games,
        "wins": wins_per_agent if args.mode != "4p_team" else [team_a_wins, team_b_wins],
        "results": results
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-a", default="submission.py")
    parser.add_argument("--agent-b", default="snapshots/test_agent.py")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--mode", choices=["2p", "4p_team", "1v3", "ffa"], default="2p")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--agents-list", nargs="+", help="Explicit list of 4 agent paths for FFA")
    args = parser.parse_args()
    
    # Multiprocessing fix for kaggle_environments
    multiprocessing.set_start_method("spawn", force=True)
    run_benchmark(args)
