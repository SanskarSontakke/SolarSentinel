import json
import math
from pathlib import Path

# Physics replication for angle-to-target reconstruction
CENTER_X = 50.0
CENTER_Y = 50.0

def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

def find_target(world_info, src_id, angle):
    # This is a bit complex without the full state, but we can guess based on distance and angle
    # For now, let's just use the diagnostic.py approach of checking if multiple players 
    # send ANY moves in the same turn to the same relative sector.
    pass

def analyze_collisions(results_path):
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    games = data.get("results", [])
    collision_turns = 0
    total_turns_with_moves = 0

    for game in games:
        move_history = game.get("move_history", {})
        for turn, p_moves in move_history.items():
            if len(p_moves) > 1:
                # Multiple players moved this turn
                collision_turns += 1
            if len(p_moves) > 0:
                total_turns_with_moves += 1

    print(f"Turns with multi-player activity: {collision_turns} / {total_turns_with_moves}")

if __name__ == "__main__":
    analyze_collisions("benchmark_results.json")
