"""
Orbit Wars — Elite Sentinel Agent v4
Target: 1500+ skill rating

Hybrid architecture (best of references 5, 6, 8):
  - Value/Cost matrix for main attack dispatch (ref 5/6 - proven at top ranks)
  - Forward simulation for defense planning (ref 6/8)
  - Iterative aiming for orbiting/comet targets (ref 3/4/6)
  - Doomed evacuation + desperado strikes (ref 5/8)
  - Supplement cooperative attacks (ref 6/8)
  - Forward funnel logistics (ref 5/6/8)
  - Finishing cleanup (ref 5/6)
  - Contested neutral avoidance (ref 5)
  - Indirect wealth scoring (ref 5/6)
"""

import logging
import warnings

# Force silence all INFO logs from kaggle_environments and absl
logging.getLogger("kaggle_environments").setLevel(logging.WARNING)
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass
logging.disable(logging.INFO)
warnings.filterwarnings("ignore")
import math
from collections import defaultdict

class Planet:
    def __init__(self, id, owner, x, y, radius, ships, production):
        self.id = id
        self.owner = owner
        self.x = x
        self.y = y
        self.radius = radius
        self.ships = ships
        self.production = production

class Fleet:
    def __init__(self, id, owner, x, y, angle, source, ships):
        self.id = id
        self.owner = owner
        self.x = x
        self.y = y
        self.angle = angle
        self.source = source
        self.ships = ships

# ═══ Constants ════════════════════════════════════════════════════════════════
CX, CY = 50.0, 50.0
SUN_R = 10.0
ROT_LIMIT = 50.0
MAX_SPEED = 6.0
TOTAL_STEPS = 500
SUN_MARGIN = 1.3
HORIZON = 80

# ═══ Physics ══════════════════════════════════════════════════════════════════

def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def _fleet_speed(ships):
    if ships <= 1:
        return 1.0
    r = math.log(max(1, min(ships, 1000))) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * max(0.0, min(1.0, r)) ** 1.5


def _crosses_sun(x1, y1, x2, y2, margin=SUN_MARGIN):
    r = SUN_R + margin
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - CX, y1 - CY
    a = dx * dx + dy * dy
    if a < 1e-9:
        return _dist(x1, y1, CX, CY) < r
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    disc = math.sqrt(disc)
    t1 = (-b - disc) / (2 * a)
    t2 = (-b + disc) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)


def _safe_angle(sx, sy, tx, ty):
    """(angle, distance) with sun-avoidance waypoints."""
    dd = _dist(sx, sy, tx, ty)
    if not _crosses_sun(sx, sy, tx, ty):
        return math.atan2(ty - sy, tx - sx), dd
    vx, vy = tx - sx, ty - sy
    nm = math.hypot(vx, vy)
    if nm < 1e-9:
        return math.atan2(ty - sy, tx - sx), dd
    nx, ny = -vy / nm, vx / nm
    best = None
    for sign in (1.0, -1.0):
        for m in (1.8, 2.3, 3.0, 4.0):
            wx = CX + sign * nx * SUN_R * m
            wy = CY + sign * ny * SUN_R * m
            if _crosses_sun(sx, sy, wx, wy) or _crosses_sun(wx, wy, tx, ty):
                continue
            dv = _dist(sx, sy, wx, wy) + _dist(wx, wy, tx, ty)
            if best is None or dv < best[0]:
                best = (dv, wx, wy)
            break
    if best is None:
        return math.atan2(ty - sy, tx - sx), dd * 1.8
    return math.atan2(best[2] - sy, best[1] - sx), best[0]


def _travel_time(sx, sy, tx, ty, ships):
    _, dv = _safe_angle(sx, sy, tx, ty)
    return max(1, int(math.ceil(dv / _fleet_speed(max(1, ships)))))


def _launch_clear(sx, sy, ang):
    lx = sx + math.cos(ang) * 3.5
    ly = sy + math.sin(ang) * 3.5
    return not _crosses_sun(sx, sy, lx, ly, 0.4)


# ═══ Orbital & comet prediction ═══════════════════════════════════════════════

def _pred_planet(planet, init_map, av, turns):
    ini = init_map.get(planet.id)
    if ini is None:
        return planet.x, planet.y
    orb = _dist(ini.x, ini.y, CX, CY)
    if orb + ini.radius >= ROT_LIMIT:
        return planet.x, planet.y
    ang = math.atan2(planet.y - CY, planet.x - CX) + av * turns
    return CX + orb * math.cos(ang), CY + orb * math.sin(ang)


def _pred_comet(pid, comets, turns):
    for g in comets:
        pids = g.get("planet_ids", [])
        if pid not in pids:
            continue
        idx = pids.index(pid)
        paths = g.get("paths", [])
        pi = g.get("path_index", 0)
        if idx >= len(paths):
            return None
        path = paths[idx]
        fi = pi + int(turns)
        return (path[fi][0], path[fi][1]) if 0 <= fi < len(path) else None
    return None


def _comet_ttl(pid, comets):
    for g in comets:
        pids = g.get("planet_ids", [])
        if pid not in pids:
            continue
        idx = pids.index(pid)
        paths = g.get("paths", [])
        pi = g.get("path_index", 0)
        return max(0, len(paths[idx]) - pi) if idx < len(paths) else 0
    return 0


def _aim_at(src, tgt, ships, init_map, av, comets, comet_ids, iters=5):
    """Iterative lead-aiming. Returns (angle, turns, tx, ty) or None."""
    tx, ty = tgt.x, tgt.y
    for _ in range(iters):
        ang, dv = _safe_angle(src.x, src.y, tx, ty)
        turns = max(1, int(math.ceil(dv / _fleet_speed(max(1, ships)))))
        if tgt.id in comet_ids:
            pos = _pred_comet(tgt.id, comets, turns)
            if pos is None:
                return None
            ntx, nty = pos
        else:
            ntx, nty = _pred_planet(tgt, init_map, av, turns)
        if abs(ntx - tx) < 0.3 and abs(nty - ty) < 0.3:
            tx, ty = ntx, nty
            break
        tx, ty = ntx, nty
    ang, dv = _safe_angle(src.x, src.y, tx, ty)
    turns = max(1, int(math.ceil(dv / _fleet_speed(max(1, ships)))))
    return ang, turns, tx, ty


# ═══ Fleet tracking ═══════════════════════════════════════════════════════════

def _fleet_dest(f, planets):
    fvx, fvy = math.cos(f.angle), math.sin(f.angle)
    sp = _fleet_speed(f.ships)
    best_p, best_t = None, 1e9
    for p in planets:
        dx, dy = p.x - f.x, p.y - f.y
        proj = dx * fvx + dy * fvy
        if proj <= 0:
            continue
        perp = abs(dx * fvy - dy * fvx)
        if perp > p.radius + 1.3:
            continue
        t = proj / sp
        if t < best_t and t <= HORIZON:
            best_t = t
            best_p = p
    return (best_p, int(math.ceil(best_t))) if best_p else (None, None)


def _build_arrivals(fleets, planets):
    arr = {p.id: [] for p in planets}
    for f in fleets:
        p, t = _fleet_dest(f, planets)
        if p:
            arr[p.id].append((t, f.owner, int(f.ships)))
    return arr


# ═══ Combat simulation ════════════════════════════════════════════════════════

def _defense_needed(planet, arrivals, player):
    """Extra ships needed so owned planet survives all incoming."""
    if planet.owner != player or not arrivals:
        return 0
    evts = sorted(arrivals, key=lambda a: a[0])
    garrison = planet.ships
    last_t = 0
    deficit = 0
    i = 0
    while i < len(evts):
        t = evts[i][0]
        garrison += (t - last_t) * planet.production
        grp = []
        while i < len(evts) and evts[i][0] == t:
            grp.append(evts[i])
            i += 1
        friendly = sum(s for _, o, s in grp if o == player)
        enemy = sum(s for _, o, s in grp if o != player)
        garrison += friendly - enemy
        if garrison < 0:
            deficit = max(deficit, -garrison + 1)
            garrison = 0
        last_t = t
    return deficit


def _safe_ships(planet, arrivals, player, horizon=HORIZON):
    """Min future garrison — maximum ships that can be safely sent."""
    if planet.owner != player:
        return 0
    if not arrivals:
        return planet.ships
    evts = sorted(arrivals, key=lambda a: a[0])
    garrison = planet.ships
    min_future = garrison
    last_t = 0
    i = 0
    while i < len(evts):
        t = evts[i][0]
        if t > horizon:
            break
        garrison += (t - last_t) * planet.production
        grp = []
        while i < len(evts) and evts[i][0] == t:
            grp.append(evts[i])
            i += 1
        friendly = sum(s for _, o, s in grp if o == player)
        enemy = sum(s for _, o, s in grp if o != player)
        garrison += friendly - enemy
        if garrison < min_future:
            min_future = garrison
        last_t = t
    return max(0, min_future)


def _simulate_future(planet, arrivals, player, horizon=HORIZON):
    """Forward-simulate planet ownership timeline."""
    if not arrivals:
        if planet.owner == -1:
            return [(0, planet.owner, planet.ships)]
        return [(horizon, planet.owner, planet.ships + planet.production * horizon)]

    evts = sorted(arrivals, key=lambda a: a[0])
    timeline = []
    garrison = planet.ships
    owner = planet.owner
    last_t = 0
    i = 0
    while i < len(evts):
        t = evts[i][0]
        if t > horizon:
            break
        if owner != -1 and t > last_t:
            garrison += (t - last_t) * planet.production
        grp = []
        while i < len(evts) and evts[i][0] == t:
            grp.append(evts[i])
            i += 1
        by_owner = {}
        for _, o, s in grp:
            by_owner[o] = by_owner.get(o, 0) + s
        if owner in by_owner:
            garrison += by_owner.pop(owner)
        attackers = sorted(by_owner.items(), key=lambda x: -x[1])
        while len(attackers) >= 2 and attackers[0][1] == attackers[1][1]:
            attackers = attackers[2:]
        if attackers:
            top_o, top_s = attackers[0]
            second = attackers[1][1] if len(attackers) > 1 else 0
            effective = top_s - second
            if effective > garrison:
                garrison = effective - garrison
                owner = top_o
            elif effective > 0:
                garrison -= effective
        last_t = t
        timeline.append((t, owner, max(0, garrison)))
    if last_t < horizon and owner != -1:
        garrison += (horizon - last_t) * planet.production
        timeline.append((horizon, owner, max(0, garrison)))
    return timeline


def _compute_defender(tgt, arrival_turns, arrivals, player):
    """Net garrison to overcome: enemy reinforces + production, minus friendly in-transit."""
    if tgt.owner == -1:
        defender = tgt.ships
    elif tgt.owner == player:
        return 0
    else:
        defender = tgt.ships + tgt.production * arrival_turns
    for t, o, s in arrivals:
        if t <= arrival_turns:
            if o == tgt.owner:
                defender += s
            elif o == player:
                defender -= s
    return max(0, int(defender))


# ═══ Scoring helpers ══════════════════════════════════════════════════════════

def _indirect_wealth(pid, planets, player):
    """Positional value: nearby production proximity."""
    tgt = None
    for p in planets:
        if p.id == pid:
            tgt = p
            break
    if tgt is None:
        return 0.0
    w = 0.0
    for p in planets:
        if p.id == tgt.id:
            continue
        d = _dist(tgt.x, tgt.y, p.x, p.y)
        if d < 1:
            continue
        factor = p.production / (d + 10.0)
        if p.owner == player:
            w += factor * 0.5
        elif p.owner == -1:
            w += factor * 1.0
        else:
            w += factor * 1.3
    return w


def _reaction_time(tgt, my_planets, enemy_planets):
    """(my_min_time, enemy_min_time) to reach target."""
    my_t = min((_travel_time(p.x, p.y, tgt.x, tgt.y, max(p.ships, 1))
                for p in my_planets), default=1e9)
    en_t = min((_travel_time(p.x, p.y, tgt.x, tgt.y, max(p.ships, 1))
                for p in enemy_planets), default=1e9)
    return my_t, en_t


# ═══ Forward Simulation ═══════════════════════════════════════════════════════

class SimFleet:
    __slots__ = ['owner', 'target_id', 'ships', 'eta']
    def __init__(self, owner, target_id, ships, eta):
        self.owner = owner
        self.target_id = target_id
        self.ships = ships
        self.eta = eta

class SimState:
    __slots__ = ['planets', 'fleets', 'geo']
    
    def __init__(self):
        self.planets = {}  # pid -> [owner, ships, production]
        self.fleets = []   # SimFleet list
        self.geo = {}      # pid -> (x, y)
        
    def clone(self):
        c = SimState()
        c.planets = {k: v.copy() for k, v in self.planets.items()}
        c.fleets = [SimFleet(f.owner, f.target_id, f.ships, f.eta) for f in self.fleets]
        c.geo = self.geo
        return c
        
    def step(self):
        for pid, pdata in self.planets.items():
            if pdata[0] != -1:
                pdata[1] += pdata[2]
                
        for f in self.fleets:
            f.eta -= 1
            
        arriving = [f for f in self.fleets if f.eta <= 0]
        self.fleets = [f for f in self.fleets if f.eta > 0]
        
        if not arriving:
            return
            
        arrivals_by_planet = defaultdict(list)
        for f in arriving:
            arrivals_by_planet[f.target_id].append(f)
            
        for pid, arriving_fleets in arrivals_by_planet.items():
            if pid not in self.planets: continue
            pdata = self.planets[pid]
            current_owner = pdata[0]
            
            forces = defaultdict(int)
            for f in arriving_fleets:
                forces[f.owner] += f.ships
            forces[current_owner] += pdata[1]
            
            if not forces: continue
            
            sorted_forces = sorted(forces.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_forces) == 1:
                pdata[0] = sorted_forces[0][0]
                pdata[1] = sorted_forces[0][1]
            else:
                rem = sorted_forces[0][1] - sorted_forces[1][1]
                pdata[1] = rem
                if rem > 0:
                    pdata[0] = sorted_forces[0][0]

def evaluate_moves(moves_list, sim_state, player, horizon=5, discount=1.0):
    state = sim_state.clone()
    
    for move in moves_list:
        if len(move) >= 5:
            src_id, ang, ships, target_id, eta = move[:5]
        else:
            src_id, ang, ships = move
            src_x, src_y = state.geo[src_id]
            target_id = None
            best_dist = 1e9
            cx, cy = math.cos(ang), math.sin(ang)
            for pid, (px, py) in state.geo.items():
                if pid == src_id: continue
                dx, dy = px - src_x, py - src_y
                dist = math.hypot(dx, dy)
                if dist < 1e-4: continue
                if (dx * cx + dy * cy) / dist > 0.999:
                    if dist < best_dist:
                        best_dist = dist
                        target_id = pid
            eta = max(1, math.ceil(best_dist / 6.0)) if target_id is not None else 1000
            
        if target_id is not None:
            state.fleets.append(SimFleet(player, target_id, ships, eta))
            if src_id in state.planets and state.planets[src_id][0] == player:
                state.planets[src_id][1] = max(0, state.planets[src_id][1] - ships)

    for _ in range(horizon):
        state.step()
        
    my_score = 0
    en_score = 0
    for pid, pdata in state.planets.items():
        owner, ships, prod = pdata
        val = ships + 40 * prod
        if owner == player:
            my_score += val
        elif owner != -1:
            en_score += val
            
    # Score ships in flight with discount factor relative to arrival time
    friendly_in_flight = 0
    for f in state.fleets:
        if f.owner == player:
            f_val = f.ships * (discount ** (f.eta / 10.0))
            friendly_in_flight += f_val
            
    enemy_in_flight = sum(f.ships for f in state.fleets if f.owner != player)
            
    return (my_score + friendly_in_flight) - (en_score + 1.0 * enemy_in_flight)

def select_best_move_set(candidates, sim_state, player, CFG):
    if not candidates:
        return []
        
    import time as _time_mod
    t0 = _time_mod.time()
    best_score = -1e9
    best_moves = candidates[0]
    
    horizon = int(CFG.get("sim_horizon", 30))
    discount = CFG.get("fleet_discount", 0.95)
    
    for i, moves_list in enumerate(candidates):
        score = evaluate_moves(moves_list, sim_state, player, horizon=horizon, discount=discount)
        if i == 0: score += 0.05
            
        if score > best_score:
            best_score = score
            best_moves = moves_list
            
        if (_time_mod.time() - t0) > 0.140: break
            
    return [[m[0], float(m[1]), int(m[2])] for m in best_moves]

# ═══ Main Agent ═══════════════════════════════════════════════════════════════

def agent(obs, override_config=None):
    CFG = {
        "enemy_multiplier": 2.442003,
        "finishing_multiplier": 2.085217,
        "early_neutral_multiplier": 1.134588,
        "safe_neutral_early_multiplier": 0.800000,
        "contested_neutral_penalty": 0.143123,
        "prod_weight": 22.728706,
        "iw_weight": 2.320458,
        "contested_margin": 2.110718,
        "cost_turns_weight": 0.106466,
        "funnel_finishing_ratio": 0.771403,
        "funnel_ratio": 0.703222,
        "sim_horizon": 25.359802,
        "fleet_discount": 0.977407,
    }
    if override_config is not None:
        CFG.update(override_config)

    import time as _time_mod
    _agent_t0 = _time_mod.time()

    # ── Parse observation ─────────────────────────────────────────────────────
    if isinstance(obs, dict): get = obs.get
    else: get = lambda k, d=None: getattr(obs, k, d)

    player    = get("player", 0)
    step      = get("step", 0) or 0
    planets   = [Planet(*p) for p in (get("planets", []) or [])]
    fleets    = [Fleet(*f)  for f in (get("fleets", []) or [])]
    av        = get("angular_velocity", 0.0) or 0.0
    init_map  = {Planet(*p).id: Planet(*p) for p in (get("initial_planets", []) or [])}
    comets    = get("comets", []) or []
    comet_ids = set(get("comet_planet_ids", []) or [])
    p_by_id   = {p.id: p for p in planets}

    sim_state = SimState()
    for p in planets:
        sim_state.planets[p.id] = [p.owner, p.ships, p.production]
        sim_state.geo[p.id] = (p.x, p.y)
    for f in fleets:
        target_p, t = _fleet_dest(f, planets)
        if target_p: sim_state.fleets.append(SimFleet(f.owner, target_p.id, f.ships, t))

    n_players = len(set(p.owner for p in planets if p.owner != -1)) + len(set(f.owner for f in fleets if f.owner != -1))
    is_4p = (n_players > 2) or (player > 1)

    mine    = [p for p in planets if p.owner == player]
    if not mine: return []
    enemy   = [p for p in planets if p.owner not in (-1, player)]
    tgts    = [p for p in planets if p.owner != player]

    rem   = max(1, TOTAL_STEPS - step)
    early = step < 40
    late  = rem < 60
    vlate = rem < 25

    # ── Macro analytics ───────────────────────────────────────────────────────
    my_ships = sum(p.ships for p in mine) + sum(int(f.ships) for f in fleets if f.owner == player)
    en_ships = sum(p.ships for p in enemy) + sum(int(f.ships) for f in fleets if f.owner not in (-1, player))
    my_prod  = sum(p.production for p in mine)
    en_prod  = sum(p.production for p in enemy)
    dom      = (my_ships - en_ships) / max(1, my_ships + en_ships)
    
    dying_player, strongest_player = -1, -1
    kingmaker_protected = set()
    vulture_targets = []

    if is_4p:
        enemy_stats = {}
        for p in enemy: enemy_stats[p.owner] = enemy_stats.get(p.owner, 0) + p.ships
        for f in fleets:
            if f.owner not in (-1, player):
                enemy_stats[f.owner] = enemy_stats.get(f.owner, 0) + int(f.ships)

        enemy_planet_owners = set(p.owner for p in enemy)
        for e_owner, e_ships in enemy_stats.items():
            if e_owner not in enemy_planet_owners and e_ships > 0:
                for f in fleets:
                    if f.owner == e_owner:
                        tp, t = _fleet_dest(f, planets)
                        if tp: vulture_targets.append(tp.id)

        sorted_enemies = sorted(enemy_stats.items(), key=lambda x: x[1])
        if sorted_enemies:
            strongest_player = sorted_enemies[-1][0]
            if len(sorted_enemies) >= 2:
                if sorted_enemies[0][1] < 0.5 * sorted_enemies[1][1]: dying_player = sorted_enemies[0][0]
                if my_ships > 1.3 * sorted_enemies[-2][1]:
                    for eo, es in sorted_enemies:
                        if es < my_ships and eo != sorted_enemies[-2][0] and eo != strongest_player:
                            kingmaker_protected.add(eo)
        finishing = dom > 0.25 and my_prod > (en_prod * 0.5) and step > 150
    else:
        finishing = dom > 0.35 and my_prod > en_prod * 1.3 and step > 100

    arr = _build_arrivals(fleets, planets)
    target_commits = {}
    planet_dispatched = {}
    
    # ── Defense & availability ────────────────────────────────────────────────
    reserve, safe_avail, doomed = {}, {}, set()
    for p in mine:
        need = _defense_needed(p, arr[p.id], player)
        reserve[p.id] = min(p.ships, need)
        safe_avail[p.id] = _safe_ships(p, arr[p.id], player)
        timeline = _simulate_future(p, arr[p.id], player)
        if any(e[1] != player and e[1] != -1 for e in timeline) and reserve[p.id] >= p.ships:
            doomed.add(p.id)

    avail = {}
    for p in mine:
        if p.id in doomed: avail[p.id] = p.ships
        elif finishing: avail[p.id] = max(0, p.ships - max(0, reserve[p.id] - 5))
        elif late: avail[p.id] = max(0, p.ships - reserve[p.id])
        else: avail[p.id] = min(safe_avail[p.id], max(0, p.ships - reserve[p.id]))

    inbound_friendly = {p.id: sum(f.ships for f in fleets if f.owner == player and _fleet_dest(f, planets)[0] == p) for p in planets}

    base_moves = []
    # ═══ Phase 1: Desperado ═══════════════════════════════════════════════════
    for sid in list(doomed):
        if avail[sid] <= 0: continue
        src = p_by_id[sid]
        best_tgt, best_score, best_ang = None, -1, None
        for tgt in tgts:
            if tgt.owner == -1: continue
            r = _aim_at(src, tgt, avail[sid], init_map, av, comets, comet_ids)
            if not r: continue
            ang, t_a, _, _ = r
            if t_a > 40 or not _launch_clear(src.x, src.y, ang): continue
            needed = _compute_defender(tgt, t_a, arr[tgt.id], player) + 1
            if avail[sid] >= needed:
                score = tgt.production / (t_a + 1.0)
                if score > best_score: best_score, best_tgt, best_ang = score, tgt, ang
        if best_tgt and best_ang is not None:
            base_moves.append([sid, float(best_ang), avail[sid]])
            if best_tgt.owner != player: target_commits[best_tgt.id] = target_commits.get(best_tgt.id, 0) + avail[sid]
            planet_dispatched[sid] = avail[sid]
            avail[sid] = 0

    # ═══ Phase 2: Value/Cost Matrix ═══════════════════════════════════════════
    candidates = []
    for src in mine:
        if avail[src.id] <= 0 or src.id in doomed: continue
        for tgt in tgts:
            if is_4p and tgt.owner in kingmaker_protected: continue
            dist = _dist(src.x, src.y, tgt.x, tgt.y)
            if dist > 140: continue
            est_t = _travel_time(src.x, src.y, tgt.x, tgt.y, max(10, avail[src.id]))
            if vlate and est_t > rem - 3: continue
            est_n = _compute_defender(tgt, est_t, arr[tgt.id], player) + 1
            if est_n > avail[src.id]: continue
            r = _aim_at(src, tgt, est_n, init_map, av, comets, comet_ids)
            if not r: continue
            ang, turns, _, _ = r
            if not _launch_clear(src.x, src.y, ang): continue
            needed = _compute_defender(tgt, turns, arr[tgt.id], player) + 1
            if needed > avail[src.id]: continue
            
            value = (tgt.production * (rem-turns) * CFG["prod_weight"]) + (_indirect_wealth(tgt.id, planets, player) * (rem-turns) * CFG["iw_weight"])
            if tgt.owner != -1:
                value *= CFG["enemy_multiplier"]
                if is_4p:
                    if tgt.owner == dying_player: value *= 0.6
                    elif tgt.owner == strongest_player: value *= 1.4
            else:
                if enemy:
                    min_e = min((_dist(tgt.x, tgt.y, e.x, e.y) for e in enemy), default=200)
                    if dist > min_e - 2: value *= CFG["contested_neutral_penalty"]
                    elif early: value *= CFG["early_neutral_multiplier"]
            
            score = value / (needed + turns * CFG["cost_turns_weight"] + 1.0)
            candidates.append((score, src.id, tgt.id, ang, needed, turns))

    candidates.sort(key=lambda x: -x[0])
    
    def build_p2(subset):
        tc, pd, m = target_commits.copy(), planet_dispatched.copy(), []
        for s, sid, tid, ang, needed, turns in subset:
            already = tc.get(tid, 0)
            missing = max(0, needed - inbound_friendly.get(tid, 0) - already)
            src_val = avail[sid] - pd.get(sid, 0)
            if missing > 0 and src_val >= missing:
                m.append([sid, ang, missing])
                pd[sid] = pd.get(sid, 0) + missing
                tc[tid] = already + missing
        return m

    greedy_m = build_p2(candidates)
    cautious_m = build_p2(candidates[:5])
    
    # Evaluate candidates
    full_sets = [base_moves + greedy_m, base_moves + cautious_m]
    best_final = select_best_move_set(full_sets, sim_state, player, CFG)
    
    return best_final
