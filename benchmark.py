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

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    plt = None
    mticker = None

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
    mode, agents_paths, game_idx, seed, verbose = args_tuple
    from kaggle_environments import make
    
    # Reload agents in child process
    agents = [load_agent(p) for p in agents_paths]
    
    log_buffer = []
    total_actions = [0] * len(agents)
    
    if verbose:
        log_buffer.append(f"========================================")
        log_buffer.append(f"=== STARTED GAME {game_idx:3d} ===")
        log_buffer.append(f"========================================")
        
    env = make("orbit_wars", debug=verbose)
    # Set seed for reproducibility in sets
    if seed is not None:
        # Note: orbit_wars might not respect a global seed easily, but we try
        pass
        
    env.run(agents)
    final = env.steps[-1]
    
    # Collect move history for diagnostics
    # Format: {turn_idx: {player_idx: [moves]}}
    move_history = {}
    
    for i, step in enumerate(env.steps):
        turn_actions = {}
        action_summaries = []
        for p_idx, player_data in enumerate(step):
            action = player_data.get("action")
            if action:
                turn_actions[p_idx] = action
                action_summaries.append(f"P{p_idx} moves: {len(action)}")
                total_actions[p_idx] += len(action)
            if player_data.get("status") == "ERROR" and verbose:
                log_buffer.append(f"[Game {game_idx}] [ERROR] Step {i}, P{p_idx} crashed! Check output above.")
        if turn_actions:
            move_history[i] = turn_actions
            
        if verbose and (turn_actions or i % 10 == 0 or i == len(env.steps) - 1):
            obs = step[0].observation
            planets = obs.planets if hasattr(obs, "planets") else obs.get("planets", [])
            ships = [0] * len(agents)
            for p in planets:
                if p[1] != -1: ships[p[1]] += p[5]
            log_buffer.append(f"[Game {game_idx}] Step {i:3d} | Ships: {ships} | Actions: {', '.join(action_summaries) if action_summaries else 'None'}")

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
    
    if verbose:
        summary_log = [f"\n--- GAME {game_idx} SUMMARY ---", f"Winner: P{winner_idx}", f"Steps:  {len(env.steps) - 1}"]
        for p_idx in range(len(agents)):
            summary_log.append(f"  P{p_idx} | Final Ships: {ship_counts[p_idx]:4d} | Total Actions: {total_actions[p_idx]}")
        summary_log.append(f"========================================\n")
        print("\n".join(summary_log))
        
    return {
        "game_idx": game_idx,
        "rewards": rewards,
        "ships": ship_counts,
        "winner": winner_idx,
        "steps": len(env.steps) - 1,
        "move_history": move_history,
        "total_actions": total_actions,
        "verbose_log": log_buffer if verbose else []
    }

def create_visualizations(results_path: Path, output_dir: Path):
    """
    Analyzes benchmark_results.json and generates insightful charts.
    """
    if plt is None:
        print("\n[WARN] Matplotlib is required for generating charts.")
        print("       Please install it using: pip install matplotlib")
        return

    if not results_path.exists():
        print(f"Error: Benchmark results not found at '{results_path}'")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    output_dir.mkdir(exist_ok=True)
    print(f"\nSaving charts to: {output_dir.resolve()}")

    num_games = data.get("games", 0)
    games = data.get("results", [])
    if not games or num_games == 0:
        print("No game results found in the file.")
        return

    num_players = len(games[0].get("rewards", []))
    player_names = [f"Player {i}" for i in range(num_players)]

    # 1. Win Rate Pie Chart
    wins = data.get("wins", [0] * num_players)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#f0f0f0')
    ax.set_facecolor('#f0f0f0')
    explode = [0.05 if w == max(wins) and max(wins) > 0 else 0 for w in wins]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_players))
    wedges, texts, autotexts = ax.pie(wins, labels=player_names, autopct=lambda p: f'{p * num_games / 100:.0f} wins\n({p:.1f}%)' if p > 0 else '', startangle=90, explode=explode, colors=colors, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    plt.setp(autotexts, size=10, weight="bold", color="white")
    plt.setp(texts, size=12)
    ax.set_title(f'Win Rate Distribution ({num_games} Games)', size=16, weight="bold", pad=20)
    plt.savefig(output_dir / "win_rate.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  - Saved win rate chart: win_rate.png")

    # 2. Average Final Ships & Actions Bar Charts
    total_ships = np.zeros(num_players)
    total_actions = np.zeros(num_players)
    for game in games:
        total_ships += np.array(game.get("ships", [0] * num_players))
        total_actions += np.array(game.get("total_actions", [0] * num_players))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#f0f0f0')
    fig.suptitle('Average Game Statistics', size=18, weight='bold')
    bars1 = ax1.bar(player_names, total_ships / num_games, color=plt.cm.plasma(np.linspace(0.2, 0.8, num_players)))
    ax1.set_ylabel('Average Ship Count'); ax1.set_title('Final Ships', size=14); ax1.bar_label(bars1, fmt='{:,.0f}'); ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}')); ax1.set_facecolor('#e9e9e9')
    bars2 = ax2.bar(player_names, total_actions / num_games, color=plt.cm.cividis(np.linspace(0.2, 0.8, num_players)))
    ax2.set_ylabel('Average Actions Sent'); ax2.set_title('Commands Issued', size=14); ax2.bar_label(bars2, fmt='{:.1f}'); ax2.set_facecolor('#e9e9e9')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "average_stats.png", dpi=150)
    plt.close(fig)
    print(f"  - Saved average stats chart: average_stats.png")

def run_benchmark(args):
    t0 = time.time()
    
    output_folder = ROOT / "benchmark_results"
    output_folder.mkdir(exist_ok=True)
    results_file = output_folder / "benchmark_results.json"

    # Clear the results file at the start to ensure clean logs
    with open(results_file, "w") as f:
        json.dump({"status": "running", "message": "Benchmark in progress..."}, f)

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
        
        jobs.append((args.mode, current_agents, i, i, args.verbose))

    # Run parallel
    with multiprocessing.Pool(args.threads) as pool:
        results = pool.map(run_single_game, jobs)

    # Process results
    wins_per_agent = [0] * len(agent_paths)
    team_a_wins, team_b_wins = 0, 0
    avg_actions = [0] * len(agent_paths)
    
    for i, res in enumerate(results):
        for p_idx, acts in enumerate(res["total_actions"]):
            avg_actions[p_idx] += acts
            
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
        
    print(f"\nAverage Actions Per Game:")
    for i in range(len(agent_paths)):
        p_name = Path(agent_paths[i]).name
        print(f"  P{i} ({p_name}): {avg_actions[i]/num_games:.1f} actions")
    
    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "games": num_games,
        "wins": wins_per_agent if args.mode != "4p_team" else [team_a_wins, team_b_wins],
        "results": results
    }
    
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
        
    create_visualizations(results_file, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-a", default="submission.py")
    parser.add_argument("--agent-b", default="snapshots/test_agent.py")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--mode", choices=["2p", "4p_team", "1v3", "ffa"], default="2p")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--agents-list", nargs="+", help="Explicit list of 4 agent paths for FFA")
    parser.add_argument("--verbose", action="store_true", help="Print per-move statistics and errors for debugging")
    args = parser.parse_args()
    
    # Multiprocessing fix for kaggle_environments
    multiprocessing.set_start_method("spawn", force=True)
    run_benchmark(args)
