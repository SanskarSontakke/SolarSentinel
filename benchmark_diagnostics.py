import json
import numpy as np
from pathlib import Path

def analyze_moves(results_path):
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    games = data.get("results", [])
    if not games:
        print("No game results found.")
        return

    player_stats = {} # player_idx -> stats

    for game in games:
        winner = game["winner"]
        move_history = game.get("move_history", {})
        
        # turn_idx is a string key from JSON
        for turn_str, turn_actions in move_history.items():
            for p_idx_str, actions in turn_actions.items():
                p_idx = int(p_idx_str)
                if p_idx not in player_stats:
                    player_stats[p_idx] = {
                        "total_ships_sent": 0,
                        "unique_targets": set(),
                        "turns_active": 0,
                        "capture_attempts": 0
                    }
                
                player_stats[p_idx]["turns_active"] += 1
                for move in actions:
                    # move format: [src_id, angle, ship_count]
                    if len(move) >= 3:
                        player_stats[p_idx]["total_ships_sent"] += move[2]
                        # We don't have target_id here directly, but we can see activity
                        player_stats[p_idx]["capture_attempts"] += 1

    print("\n=== Aggression Analysis ===")
    for p_idx, stats in sorted(player_stats.items()):
        name = "SolarSentinel" if p_idx == 0 else f"Basline_{p_idx}"
        avg_ships = stats["total_ships_sent"] / len(games)
        avg_cmds = stats["capture_attempts"] / len(games)
        print(f"Player {p_idx} ({name}):")
        print(f"  Avg Ships Sent: {avg_ships:.1f}")
        print(f"  Avg Commands:   {avg_cmds:.1f}")

if __name__ == "__main__":
    analyze_moves("benchmark_results.json")
