# Orbit Wars — Advanced Agent Strategy Guide
**Target: 1000+ leaderboard score**

---

## Files

| File | Purpose |
|------|---------|
| `submission.py` | **Main submission** — pure Python, no extra deps |
| `train_rl.py` | PPO self-play RL trainer (TPU/GPU/CPU) |
| `evaluate.py` | Local benchmark harness |

---

## Architecture of `submission.py`

The agent runs **6 strategic phases** per turn, all within 1 second:

```
Phase 1 ─ Candidate scoring          (greedy expansion)
Phase 2 ─ Multi-source coordination  (cooperative attacks)
Phase 3 ─ Doomed planet evacuation   (don't waste ships)
Phase 4 ─ Rear-line consolidation    (forward surplus ships)
Phase 5 ─ Finishing blitz            (pile on when winning)
Phase 6 ─ Comet rush                 (instant comet capture)
```

### Key improvements over starter kit

| Feature | Starter Kit | Our Agent |
|---------|-------------|-----------|
| Planet targeting | Nearest only | Value-scored with production × time |
| Orbiting planets | Ignored | Iterative intercept prediction |
| Comets | Ignored | Actively hunted |
| Defense | None | Garrison shortfall simulation |
| Coordination | None | Cross-planet commitment tracking |
| Sun routing | Basic | Perpendicular waypoint bypass |
| Game phases | None | Early/Mid/Late/Finishing/Desperate |
| Doomed planets | Ships wasted | Evacuated to allies |
| Rear consolidation | None | Ships forwarded to front |

---

## Strategy Breakdown

### Early Game (steps 0–45)
- **Aggressive expansion**: bonus score for small fleets to neutral planets
- Grab 2–4 neutral planets before the enemy
- Prioritize high-production planets (prod 4–5)

### Mid Game (steps 45–420)
- **Coordinated attacks**: if one planet can't take a target alone, split from multiple
- **Defense simulation**: model all incoming fleets, reserve enough garrison
- **Comet opportunism**: always grab comets that have enough life remaining
- **Rear consolidation**: rear planets funnel ships to front-line planets

### Late Game (steps 420+)
- **Ship counting**: track `(my_ships - enemy_ships) / total`
- If domination > 0.38: **finishing mode** (aggressive, attack weakest enemy)
- If domination < -0.30: **desperate mode** (attack enemy instead of neutral)
- Never send ships that arrive after the game ends

---

## How Scoring Works

The target value function:

```
value(tgt, arrival_turns) =
    tgt.production × (remaining_turns - arrival_turns)   # base profit
    × position_bonus                                       # closer to enemy = strategic
    × 2.1  if enemy planet                                # capturing enemy is crucial
    × 1.5  if early + neutral                             # expansion premium
    × 0.5  if enemy gets there first                      # contested discount
```

Candidate score:
```
score = value / (ships_cost + 0.4 × turns)
```

---

## RL Training (optional upgrade path)

The `train_rl.py` implements PPO self-play with:
- **State encoder**: planet features + fleet features + global scalars → 512-dim MLP
- **Policy head**: selects (planet, ship_fraction) pairs
- **Value head**: estimates game outcome
- **Self-play**: trains against the rule-based agent as baseline

### Hardware priority
1. **Kaggle TPU VM** (fastest, free):
   ```bash
   python train_rl.py --backend tpu --steps 2000000
   ```
2. **P100 GPU** (Kaggle / Colab):
   ```bash
   python train_rl.py --backend gpu --steps 1000000
   ```
3. **Local CPU** (debugging):
   ```bash
   python train_rl.py --backend cpu --steps 50000 --debug
   ```

---

## Parameter Tuning Cheat Sheet

Tweak these in `submission.py` to adjust behavior:

```python
DOMINATION_HI  = 0.38   # threshold to enter finishing mode (lower = more aggressive)
DOMINATION_LO  = -0.30  # threshold to enter desperate mode
HORIZON        = 90     # planning horizon in turns (higher = more strategic)
SUN_SAFETY     = 2.0    # sun avoidance buffer (higher = safer but longer routes)
EARLY_THRESH   = 45     # turn where early game ends
LATE_THRESH    = 80     # remaining turns where late game starts
```

---

## Quick Start

```bash
# Install dependency
pip install kaggle-environments

# Test locally
python evaluate.py --games 30 --opponent nearest_sniper

# Test 4-player
python evaluate.py --games 20 --4p

# Submit
kaggle competitions submit -c orbit-wars -f submission.py -m "Advanced heuristic v1"
```

---

## Expected Performance

| Opponent | Expected Win Rate |
|----------|------------------|
| Random agent | ~98% |
| Nearest sniper | ~85-90% |
| Mid-tier bots (μ≈700) | ~60-70% |
| Top-tier bots (μ≈950) | ~45-55% |

Starting μ₀ = 600. Each win against similar-rated bots gains ~15–25 μ.
To reach 1000: need to beat ~80% of bots rated below 1000.
