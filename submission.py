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


# ═══ Main Agent ═══════════════════════════════════════════════════════════════

def agent(obs, override_config=None):
    CFG = {
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
    if override_config is not None:
        CFG.update(override_config)

    # ── Parse observation ─────────────────────────────────────────────────────
    if isinstance(obs, dict):
        get = obs.get
    else:
        get = lambda k, d=None: getattr(obs, k, d)

    player    = get("player", 0)
    step      = get("step", 0) or 0
    planets   = [Planet(*p) for p in (get("planets", []) or [])]
    fleets    = [Fleet(*f)  for f in (get("fleets", []) or [])]
    av        = get("angular_velocity", 0.0) or 0.0
    init_map  = {Planet(*p).id: Planet(*p) for p in (get("initial_planets", []) or [])}
    comets    = get("comets", []) or []
    comet_ids = set(get("comet_planet_ids", []) or [])
    p_by_id   = {p.id: p for p in planets}

    mine    = [p for p in planets if p.owner == player]
    if not mine:
        return []
    enemy   = [p for p in planets if p.owner not in (-1, player)]
    neutral = [p for p in planets if p.owner == -1]
    tgts    = [p for p in planets if p.owner != player]

    rem   = max(1, TOTAL_STEPS - step)
    early = step < 40
    mid_early = step < 80
    late  = rem < 60
    vlate = rem < 25

    # ── Macro analytics ───────────────────────────────────────────────────────
    my_ships = sum(p.ships for p in mine) + sum(int(f.ships) for f in fleets if f.owner == player)
    en_ships = sum(p.ships for p in enemy) + sum(int(f.ships) for f in fleets if f.owner not in (-1, player))
    my_prod  = sum(p.production for p in mine)
    en_prod  = sum(p.production for p in enemy)
    dom      = (my_ships - en_ships) / max(1, my_ships + en_ships)
    finishing = dom > 0.35 and my_prod > en_prod * 1.3 and step > 100
    behind   = dom < -0.25

    # ── Fleet arrivals ────────────────────────────────────────────────────────
    arr = _build_arrivals(fleets, planets)

    # ── Defense & availability ────────────────────────────────────────────────
    reserve = {}
    safe_avail = {}
    doomed = set()

    for p in mine:
        need = _defense_needed(p, arr[p.id], player)
        reserve[p.id] = min(p.ships, need)
        safe_avail[p.id] = _safe_ships(p, arr[p.id], player)

        # Check if planet is doomed (will be lost regardless)
        timeline = _simulate_future(p, arr[p.id], player)
        will_be_lost = any(e[1] != player and e[1] != -1 for e in timeline)
        if will_be_lost and reserve[p.id] >= p.ships:
            doomed.add(p.id)

    # Available ships for offence
    avail = {}
    for p in mine:
        if p.id in doomed:
            avail[p.id] = p.ships
        elif finishing:
            avail[p.id] = max(0, p.ships - max(0, reserve[p.id] - 5))
        elif late:
            avail[p.id] = max(0, p.ships - reserve[p.id])
        else:
            avail[p.id] = min(safe_avail[p.id], max(0, p.ships - reserve[p.id]))

    # ── In-flight tracking ────────────────────────────────────────────────────
    inbound_friendly = {}
    inbound_enemy = {}
    for f in fleets:
        p, _ = _fleet_dest(f, planets)
        if p is None:
            continue
        if f.owner == player:
            inbound_friendly[p.id] = inbound_friendly.get(p.id, 0) + int(f.ships)
        else:
            inbound_enemy[p.id] = inbound_enemy.get(p.id, 0) + int(f.ships)

    # ── Dispatch state ────────────────────────────────────────────────────────
    target_commits = {}  # ships committed to each target this turn
    planet_dispatched = {}  # ships dispatched from each source
    moves = []

    # ═══ PHASE 1: Desperado strikes from doomed planets ═══════════════════════
    for sid in list(doomed):
        if avail[sid] <= 0:
            continue
        src = p_by_id[sid]
        best_tgt, best_score, best_ang = None, -1, None

        # Try attacking enemy planets
        for tgt in tgts:
            if tgt.owner == -1:
                continue
            r = _aim_at(src, tgt, avail[sid], init_map, av, comets, comet_ids)
            if r is None:
                continue
            ang, t_a, _, _ = r
            if t_a > 40 or not _launch_clear(src.x, src.y, ang):
                continue
            needed = _compute_defender(tgt, t_a, arr[tgt.id], player) + 1
            if avail[sid] >= needed:
                score = tgt.production / (t_a + 1.0)
                if score > best_score:
                    best_score, best_tgt, best_ang = score, tgt, ang

        # Fallback: retreat to nearest safe ally
        if best_tgt is None:
            safe_allies = [p for p in mine if p.id not in doomed and p.id != sid]
            if safe_allies:
                closest = min(safe_allies, key=lambda p: _dist(src.x, src.y, p.x, p.y))
                r = _aim_at(src, closest, avail[sid], init_map, av, comets, comet_ids)
                if r:
                    ang, t_a, _, _ = r
                    if _launch_clear(src.x, src.y, ang):
                        best_tgt, best_ang = closest, ang

        if best_tgt and best_ang is not None:
            moves.append([sid, float(best_ang), avail[sid]])
            if best_tgt.owner != player:
                target_commits[best_tgt.id] = target_commits.get(best_tgt.id, 0) + avail[sid]
            planet_dispatched[sid] = avail[sid]
            avail[sid] = 0

    # ═══ PHASE 2: Value/Cost Matrix — main attack dispatch ════════════════════
    candidates = []
    for src in mine:
        if avail[src.id] <= 0 or src.id in doomed:
            continue
        src_ships = avail[src.id]

        for tgt in tgts:
            my_dist = _dist(src.x, src.y, tgt.x, tgt.y)
            if my_dist > 140:
                continue

            # Quick estimate
            est_turns = _travel_time(src.x, src.y, tgt.x, tgt.y, max(10, src_ships))
            if vlate and est_turns > rem - 3:
                continue
            if tgt.id in comet_ids and est_turns >= _comet_ttl(tgt.id, comets):
                continue

            # Compute defender strength
            est_needed = _compute_defender(tgt, est_turns, arr[tgt.id], player) + 1
            if est_needed > src_ships:
                continue

            # Iterative aiming
            r = _aim_at(src, tgt, est_needed, init_map, av, comets, comet_ids)
            if r is None:
                continue
            ang, turns, _, _ = r
            if not _launch_clear(src.x, src.y, ang):
                continue

            ships_needed = _compute_defender(tgt, turns, arr[tgt.id], player) + 1
            if ships_needed > src_ships:
                continue

            # Contested neutral check
            if tgt.owner == -1 and enemy:
                my_t, en_t = _reaction_time(tgt, mine, enemy)
                if en_t <= turns:
                    ships_needed = min(src_ships, int(ships_needed * CFG["contested_margin"]) + 3)

            # ── Value computation ──
            turns_profit = max(1, rem - turns)
            if tgt.id in comet_ids:
                life = _comet_ttl(tgt.id, comets)
                turns_profit = max(0, min(turns_profit, life - turns))
                if turns_profit <= 0:
                    continue

            iw = _indirect_wealth(tgt.id, planets, player)
            value = (tgt.production * turns_profit * CFG["prod_weight"]) + (iw * turns_profit * CFG["iw_weight"])

            if tgt.owner != -1:
                # Enemy planet: double value (gain + deny)
                value *= CFG["enemy_multiplier"]
                if finishing:
                    value *= CFG["finishing_multiplier"]
                if tgt.ships < 15:
                    value += 500
            else:
                # Neutral: contested avoidance
                if enemy:
                    min_e_dist = min((_dist(tgt.x, tgt.y, e.x, e.y) for e in enemy), default=1e9)
                    if my_dist > min_e_dist - 2:
                        value *= CFG["contested_neutral_penalty"]  # Enemy is closer — let them waste ships
                    elif early:
                        value *= CFG["early_neutral_multiplier"]   # Safe early grab
                else:
                    if early:
                        value *= CFG["safe_neutral_early_multiplier"]

            cost = ships_needed + turns * CFG["cost_turns_weight"]
            score = value / (cost + 1.0)

            # Bonuses
            if early and tgt.owner == -1 and ships_needed <= 18:
                score *= 1.6
            if tgt.id in comet_ids and tgt.production >= 1:
                score *= 1.1

            candidates.append((score, src.id, tgt.id, ang, ships_needed, turns, value))

    # ── Greedy dispatch from sorted matrix ────────────────────────────────────
    candidates.sort(key=lambda x: -x[0])

    for score, sid, tid, ang, ships_needed, turns, value in candidates:
        already = target_commits.get(tid, 0)
        tgt = p_by_id[tid]
        base_needed = _compute_defender(tgt, turns, arr[tgt.id], player) + 1
        missing = max(0, base_needed - inbound_friendly.get(tid, 0) - already)
        if missing <= 0:
            continue

        src_avail = avail[sid] - planet_dispatched.get(sid, 0)
        if src_avail <= 0:
            continue

        send = min(src_avail, missing)
        if send < 1:
            continue

        moves.append([sid, float(ang), int(send)])
        planet_dispatched[sid] = planet_dispatched.get(sid, 0) + send
        target_commits[tid] = already + send

    # Update avail
    for sid in planet_dispatched:
        avail[sid] = max(0, avail[sid] - planet_dispatched.get(sid, 0))

    # ═══ PHASE 3: Supplement attacks (cooperative gap-filling) ═════════════════
    if not vlate:
        for src in mine:
            if avail[src.id] < 8 or src.id in doomed:
                continue
            src_ships = avail[src.id]
            best = None
            for tgt in tgts:
                d0 = _dist(src.x, src.y, tgt.x, tgt.y)
                if d0 > 140:
                    continue
                est_turns = _travel_time(src.x, src.y, tgt.x, tgt.y, max(10, src_ships))
                if late and est_turns > rem - 5:
                    continue
                if tgt.id in comet_ids and est_turns >= _comet_ttl(tgt.id, comets):
                    continue
                defender = _compute_defender(tgt, est_turns, arr[tgt.id], player)
                committed = target_commits.get(tgt.id, 0) + inbound_friendly.get(tgt.id, 0)
                missing = max(0, defender + 1 - committed)
                if missing <= 0:
                    continue
                if missing > src_ships and committed == 0:
                    continue
                send = min(src_ships, missing)
                if send < 5:
                    continue
                if committed + send < defender + 1:
                    continue
                turns_profit = max(1, rem - est_turns)
                v = tgt.production * turns_profit
                if tgt.owner != -1:
                    v *= 2.0
                score_s = v / (send + est_turns * 0.5 + 1.0)
                if best is None or score_s > best[0]:
                    best = (score_s, tgt, send, est_turns)

            if best is None:
                continue
            _, tgt, send, est_turns = best
            r = _aim_at(src, tgt, send, init_map, av, comets, comet_ids)
            if r is None:
                continue
            ang, turns, _, _ = r
            if not _launch_clear(src.x, src.y, ang):
                continue
            moves.append([src.id, float(ang), int(send)])
            avail[src.id] -= send
            target_commits[tgt.id] = target_commits.get(tgt.id, 0) + send

    # ═══ PHASE 4: Forward funnel (rear → front logistics) ═════════════════════
    if not vlate and len(mine) > 1 and (enemy or neutral):
        ref_set = enemy if enemy else neutral
        front_dist = {p.id: min((_dist(p.x, p.y, e.x, e.y) for e in ref_set), default=200)
                      for p in mine}
        front = min(mine, key=lambda p: front_dist[p.id])

        # Score each friendly planet by proximity to enemy production
        front_scores = {}
        if enemy:
            front_scores = {p.id: sum(e.production / (_dist(p.x, p.y, e.x, e.y) + 1.0) for e in enemy)
                            for p in mine}
            best_front = max(mine, key=lambda p: front_scores.get(p.id, 0))
        else:
            best_front = front

        send_ratio = CFG["funnel_finishing_ratio"] if finishing else CFG["funnel_ratio"]

        for r in sorted(mine, key=lambda p: -front_dist.get(p.id, 0)):
            if r.id == best_front.id or r.id in doomed:
                continue
            if front_dist.get(r.id, 0) < front_dist.get(best_front.id, 0) * 1.2:
                continue
            if avail[r.id] < 15:
                continue

            # Find forward target (closer ally)
            mid = [p for p in mine if p.id != r.id and p.id not in doomed
                   and front_dist.get(p.id, 1e9) < front_dist.get(r.id, 0) * 0.75]
            if mid:
                mid.sort(key=lambda p: _dist(r.x, r.y, p.x, p.y))
                fwd_tgt = mid[0]
            else:
                fwd_tgt = best_front
            if fwd_tgt.id == r.id:
                continue

            send = int(avail[r.id] * send_ratio)
            if send < 10:
                continue

            ra = _aim_at(r, fwd_tgt, send, init_map, av, comets, comet_ids)
            if ra is None:
                continue
            ang, turns, _, _ = ra
            if turns > 40 or not _launch_clear(r.x, r.y, ang):
                continue
            moves.append([r.id, float(ang), int(send)])
            avail[r.id] -= send

    # ═══ PHASE 5: Finishing cleanup ═══════════════════════════════════════════
    if finishing and enemy:
        weak = sorted(enemy, key=lambda p: p.ships + p.production * 10)
        for src in mine:
            if avail[src.id] < 25:
                continue
            for tgt in weak[:3]:
                if target_commits.get(tgt.id, 0) + inbound_friendly.get(tgt.id, 0) > tgt.ships + tgt.production * 5:
                    continue
                d0 = _dist(src.x, src.y, tgt.x, tgt.y)
                if d0 > 120:
                    continue
                turns_est = _travel_time(src.x, src.y, tgt.x, tgt.y, avail[src.id])
                if turns_est > rem - 5:
                    continue
                needed = _compute_defender(tgt, turns_est, arr[tgt.id], player) + 1
                committed = target_commits.get(tgt.id, 0) + inbound_friendly.get(tgt.id, 0)
                missing = max(0, needed - committed)
                if missing <= 0:
                    continue
                send = min(avail[src.id], missing + 5)
                if send < 10:
                    continue
                ra = _aim_at(src, tgt, send, init_map, av, comets, comet_ids)
                if ra is None:
                    continue
                ang, turns, _, _ = ra
                if not _launch_clear(src.x, src.y, ang):
                    continue
                moves.append([src.id, float(ang), int(send)])
                avail[src.id] -= send
                target_commits[tgt.id] = target_commits.get(tgt.id, 0) + send
                break

    # ═══ FINAL: Deduplicate & validate ════════════════════════════════════════
    dedup = {}
    for sid, ang, sh in moves:
        key = (sid, round(ang, 4))
        if key in dedup:
            dedup[key] = (sid, ang, dedup[key][2] + sh)
        else:
            dedup[key] = (sid, ang, sh)

    used = {}
    final = []
    for sid, ang, sh in dedup.values():
        src = p_by_id.get(sid)
        if src is None:
            continue
        mx = src.ships - used.get(sid, 0)
        send = min(sh, mx)
        if send >= 1:
            final.append([sid, float(ang), int(send)])
            used[sid] = used.get(sid, 0) + send

    return final
