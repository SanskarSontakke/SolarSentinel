"""
SolarSentinel — Standalone RL Training Test (Take 3)
Includes FULL formal registration of the 'orbit_wars' environment for Kaggle workers.
"""

import os, math, json, argparse, time, random
from collections import deque, defaultdict, namedtuple
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ── Install kaggle-environments if missing ──
try:
    import kaggle_environments
    from kaggle_environments import make, register
except ImportError:
    os.system("pip install kaggle-environments")
    import kaggle_environments
    from kaggle_environments import make, register

# ==============================================================================
# ── ORBIT WARS FULL SPECIFICATION ───────────────────────────────────────────
# ==============================================================================

# Constants from orbit_wars.py
BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
PLANET_CLEARANCE = 7
MIN_PLANET_GROUPS = 5
MAX_PLANET_GROUPS = 10
MIN_STATIC_GROUPS = 3
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet = namedtuple("Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def point_to_segment_distance(p, v, w):
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0: return distance(p, v)
    t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return distance(p, projection)

def generate_planets():
    planets = []
    num_q1 = random.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS)
    id_counter = 0; static_groups = 0
    for _ in range(5000):
        if static_groups >= MIN_STATIC_GROUPS: break
        prod = random.randint(1, 5)
        r = 1 + math.log(prod)
        angle = random.uniform(0, math.pi / 2)
        min_orbital = ROTATION_RADIUS_LIMIT - r
        max_orbital = (BOARD_SIZE - CENTER - r) / max(math.cos(angle), math.sin(angle))
        if min_orbital > max_orbital: continue
        orbital_r = random.uniform(min_orbital, max_orbital)
        x = CENTER + orbital_r * math.cos(angle)
        y = CENTER + orbital_r * math.sin(angle)
        if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0: continue
        if (x - CENTER) < r + 5 or (y - CENTER) < r + 5: continue
        ships = min(random.randint(5, 99), random.randint(5, 99))
        temp = [[id_counter, -1, x, y, r, ships, prod],
                [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
                [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
                [id_counter + 3, -1, BOARD_SIZE - x, BOARD_SIZE - y, r, ships, prod]]
        valid = True
        for tp in temp:
            for p in planets:
                if distance((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False; break
            if not valid: break
        if valid:
            planets.extend(temp); id_counter += 4; static_groups += 1

    for _ in range(1000):
        prod = random.randint(1, 5); r = 1 + math.log(prod)
        min_orbital = SUN_RADIUS + r + 10; max_orbital = ROTATION_RADIUS_LIMIT - r
        if min_orbital >= max_orbital: continue
        orbital_r = random.uniform(min_orbital, max_orbital)
        x = CENTER + orbital_r * math.cos(math.pi / 4)
        y = CENTER + orbital_r * math.sin(math.pi / 4)
        ships = min(random.randint(5, 99), random.randint(5, 99))
        temp = [[id_counter, -1, x, y, r, ships, prod],
                [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
                [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
                [id_counter + 3, -1, BOARD_SIZE - x, BOARD_SIZE - y, r, ships, prod]]
        valid = True
        for tp in temp:
            tp_orb = distance((tp[2], tp[3]), (CENTER, CENTER))
            for p in planets:
                p_orb = distance((p[2], p[3]), (CENTER, CENTER))
                if distance((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False; break
                if p_orb + p[4] >= ROTATION_RADIUS_LIMIT and abs(tp_orb - p_orb) < tp[4] + p[4] + PLANET_CLEARANCE:
                    valid = False; break
            if not valid: break
        if valid: planets.extend(temp); id_counter += 4; break
    return planets

def get(d, key, default):
    if isinstance(d, dict): return d.get(key, default)
    return getattr(d, key, default)

def interpreter(state, env):
    num_agents = len(state); obs0 = state[0].observation
    if env.done: return state
    if not hasattr(obs0, "planets") or not obs0.planets:
        obs0.angular_velocity = random.uniform(0.025, 0.05)
        obs0.planets = generate_planets()
        obs0.initial_planets = [p.copy() for p in obs0.planets]
        obs0.fleets = []; obs0.next_fleet_id = 0; obs0.comets = []; obs0.comet_planet_ids = []
        home = (random.randint(0, len(obs0.planets)//4 - 1)) * 4
        if num_agents == 2:
            obs0.planets[home][1] = 0; obs0.planets[home][5] = 10
            obs0.planets[home+3][1] = 1; obs0.planets[home+3][5] = 10
        for i in range(num_agents): 
            state[i].observation.player = i
            if i > 0:
                for k in ["angular_velocity", "planets", "initial_planets", "fleets", "next_fleet_id", "comets", "comet_planet_ids"]:
                    setattr(state[i].observation, k, getattr(obs0, k))
        return state

    for i in range(num_agents):
        act = state[i].action
        if act and isinstance(act, list):
            for move in act:
                if len(move) == 3:
                    fid, ang, ships = move; ships = int(ships)
                    p = next((p for p in obs0.planets if p[0] == fid), None)
                    if p and p[1] == i and p[5] >= ships and ships > 0:
                        p[5] -= ships
                        sx, sy = p[2] + math.cos(ang)*(p[4]+0.1), p[3] + math.sin(ang)*(p[4]+0.1)
                        obs0.fleets.append([obs0.next_fleet_id, i, sx, sy, ang, fid, ships])
                        obs0.next_fleet_id += 1

    for p in obs0.planets: 
        if p[1] != -1: p[5] += p[6]
    
    mspeed = env.configuration.get("shipSpeed", 6.0)
    to_rem = []; combat = defaultdict(list)
    for f in obs0.fleets:
        ships, ang = f[6], f[4]
        spd = 1.0 + (mspeed-1.0) * (math.log(ships)/math.log(1000))**1.5
        f[2] += math.cos(ang)*min(spd, mspeed); f[3] += math.sin(ang)*min(spd, mspeed)
        if not (0 <= f[2] <= 100 and 0 <= f[3] <= 100) or distance((f[2], f[3]), (50, 50)) < 10:
            to_rem.append(f); continue
        for p in obs0.planets:
            if distance((p[2], p[3]), (f[2], f[3])) < p[4]:
                combat[p[0]].append(f); to_rem.append(f); break
    
    step = get(obs0, "step", 1)
    for p in obs0.planets:
        ip = next((i for i in obs0.initial_planets if i[0] == p[0]), None)
        if ip:
            dx, dy = ip[2]-50, ip[3]-50; r = math.hypot(dx, dy)
            if r + p[4] < 50:
                ang = math.atan2(dy, dx) + obs0.angular_velocity * step
                p[2], p[3] = 50 + r*math.cos(ang), 50 + r*math.sin(ang)
    
    obs0.fleets = [f for f in obs0.fleets if f not in to_rem]
    for pid, fleets in combat.items():
        p = next((pl for pl in obs0.planets if pl[0] == pid), None)
        if p:
            sp = defaultdict(int)
            for f in fleets: sp[f[1]] += f[6]
            best = max(sp, key=sp.get); delta = sp[best]
            if p[1] == best: p[5] += delta
            else:
                p[5] -= delta
                if p[5] < 0: p[1] = best; p[5] = abs(p[5])

    if step >= env.configuration.get("episodeSteps", 500) - 2:
        for s in state: s.status = "DONE"
        sc = defaultdict(int)
        for p in obs0.planets: 
            if p[1] != -1: sc[p[1]] += p[5]
        mx = max(sc.values()) if sc else 0
        for i in range(num_agents): state[i].reward = 1 if sc[i] == mx and mx > 0 else -1
    return state

def renderer(state, env):
    obs = state[0].observation
    out = f"Step {get(obs, 'step', 0)}\nPlanets:\n"
    for p in get(obs, "planets", []):
        out += f"  ID: {p[0]}, Owner: {p[1]}, Ships: {p[5]}\n"
    return out

def html_renderer(): return ""

ORBIT_WARS_SPEC = {
    "name": "orbit_wars", "title": "Orbit Wars", "version": "1.0.9", "agents": [2, 4],
    "configuration": {
        "episodeSteps": {"type": "integer", "default": 500},
        "actTimeout": {"type": "number", "default": 6},
        "runTimeout": {"type": "number", "default": 1200},
        "shipSpeed": {"type": "number", "default": 6.0},
        "cometSpeed": {"type": "number", "default": 4.0}
    },
    "reward": {"type": "number", "default": 0},
    "observation": {
        "planets": {"type": "array", "default": []},
        "fleets": {"type": "array", "default": []},
        "player": {"type": "integer", "default": 0},
        "angular_velocity": {"type": "number", "default": 0}
    },
    "action": {"type": "array", "default": []}
}

# ── Manual registration ──
try:
    make("orbit_wars")
except Exception:
    register("orbit_wars", {
        "interpreter": interpreter, 
        "specification": ORBIT_WARS_SPEC,
        "renderer": renderer,
        "html_renderer": html_renderer
    })

# ==============================================================================
# ── TRAINING SCRIPT ───────────────────────────────────────────────────────────
# ==============================================================================

def heuristic_agent(obs):
    p = obs.player; planets = [Planet(*pl) for pl in obs.planets]
    mine = [pl for pl in planets if pl.owner == p]
    tgts = [pl for pl in planets if pl.owner != p]
    if not mine or not tgts: return []
    moves = []
    for m in mine:
        if m.ships < 10: continue
        t = min(tgts, key=lambda tx: math.hypot(m.x - tx.x, m.y - tx.y))
        moves.append([m.id, float(math.atan2(t.y - m.y, t.x - m.x)), int(m.ships * 0.5)])
    return moves

STATE_DIM = 48 * 7 + 14
class OrbitWarsNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
    def forward(self, x): return self.fc(x)

def train():
    device = torch.device("cpu")
    print(f"Device: {device}")
    net = OrbitWarsNet(STATE_DIM).to(device)
    env = make("orbit_wars", debug=False)
    steps = 0; max_steps = 10000
    print("Starting Training Smoke Test (Take 3)...")
    while steps < max_steps:
        env.reset()
        env.run([heuristic_agent, heuristic_agent])
        steps += env.steps[-1][0].observation.step
        if steps % 2000 == 0 or steps >= max_steps:
            print(f"Step {steps}/{max_steps} | Simulated rewards logging...")
    print("Training Test Complete!")
    torch.save(net.state_dict(), "model_test.pt")

if __name__ == "__main__":
    train()
