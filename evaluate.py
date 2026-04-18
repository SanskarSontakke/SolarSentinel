"""
Local evaluation harness for the Orbit Wars agent.
Run this before submitting to check your agent beats the baselines.

Usage:
  pip install kaggle-environments
  python evaluate.py --games 50 --opponent random
  python evaluate.py --games 20 --opponent nearest_sniper
  python evaluate.py --games 20 --4p
"""

import argparse, math, sys, time
from collections import defaultdict

def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

# ── baseline opponents ──────────────────────────────────────────────

def random_agent(obs):
    import random, math
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    my_planets = [p for p in planets if p[1] == player]
    moves = []
    for p in my_planets:
        if p[5] > 10:
            angle = random.uniform(0, 2 * math.pi)
            ships = random.randint(5, p[5])
            moves.append([p[0], angle, ships])
    return moves

def nearest_sniper(obs):
    import math
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    from collections import namedtuple
    Planet = namedtuple("P", ["id","owner","x","y","radius","ships","production"])
    planets = [Planet(*p) for p in raw]
    my_pl  = [p for p in planets if p.owner == player]
    tgts   = [p for p in planets if p.owner != player]
    moves  = []
    for mine in my_pl:
        if not tgts:
            break
        nearest = min(tgts, key=lambda t: math.hypot(mine.x-t.x, mine.y-t.y))
        needed  = max(nearest.ships + 1, 20)
        if mine.ships >= needed:
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, needed])
    return moves

OPPONENTS = {
    "random":         random_agent,
    "nearest_sniper": nearest_sniper,
}

# ── evaluation ──────────────────────────────────────────────────────

def run_eval(args):
    try:
        from kaggle_environments import make
    except ImportError:
        print("Install kaggle-environments: pip install kaggle-environments")
        sys.exit(1)

    import importlib.util
    spec = importlib.util.spec_from_file_location("submission", "submission.py")
    sub  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    our_agent = sub.agent

    opponent  = OPPONENTS.get(args.opponent, random_agent)
    n_players = 4 if args.four_player else 2

    wins = losses = draws = 0
    total_score_diff = 0.0
    t0 = time.time()

    for g in range(args.games):
        env = make("orbit_wars", debug=False)
        if n_players == 2:
            env.run([our_agent, opponent])
        else:
            env.run([our_agent, opponent, opponent, opponent])

        final = env.steps[-1]
        our_r = final[0].reward

        if our_r is None:
            draws += 1
        elif our_r > 0:
            wins += 1
        else:
            losses += 1

        # approximate score from ship counts
        obs = env.steps[-1][0].observation
        all_pl = obs.planets
        our_ships = sum(p[5] for p in all_pl if p[1] == 0)
        en_ships  = sum(p[5] for p in all_pl if p[1] != 0 and p[1] != -1)
        total_score_diff += (our_ships - en_ships)

        if (g + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"Game {g+1}/{args.games} | "
                  f"W={wins} L={losses} D={draws} | "
                  f"WR={wins/(g+1)*100:.1f}% | "
                  f"AvgShipLead={total_score_diff/(g+1):.0f} | "
                  f"{elapsed:.1f}s")

    total = args.games
    print(f"\n{'='*55}")
    print(f"Final Results vs '{args.opponent}' ({n_players}P):")
    print(f"  W={wins}  L={losses}  D={draws}  ({total} games)")
    print(f"  Win Rate:       {wins/total*100:.1f}%")
    print(f"  Avg Ship Lead:  {total_score_diff/total:.0f}")
    print(f"{'='*55}")
    print()
    if wins / total >= 0.75:
        print("✓ Agent looks strong. Ready to submit!")
    elif wins / total >= 0.55:
        print("~ Decent. Consider tuning parameters further.")
    else:
        print("✗ Win rate low. Review strategy logic.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games",    type=int,  default=20)
    parser.add_argument("--opponent", default="nearest_sniper",
                        choices=list(OPPONENTS.keys()))
    parser.add_argument("--4p",       action="store_true", dest="four_player")
    args = parser.parse_args()
    run_eval(args)
