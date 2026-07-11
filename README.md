# SolarSentinel
> A strategic agent for the Kaggle Orbit Wars competition, using rule-based heuristics and optional reinforcement learning.

## What it does

SolarSentinel competes in Kaggle's Orbit Wars game—a real-time space strategy competition where agents control ships, capture planets, and dodge a deadly sun. The agent runs six strategic phases per turn (planet scoring, coordination, evacuation, consolidation, finishing, comet capture) all within a 1-second time limit, and includes optional self-play reinforcement learning for performance optimization.

## Why I built it

This was a learning project to explore competitive agent design: combining rule-based heuristics with simulation, reinforcement learning optimization, and empirical benchmarking against different strategies.

## Tech stack

- Python 3 (pure stdlib submission; trainer uses NumPy)
- Kaggle Environments
- Optional: PyTorch PPO for self-play training

## Getting started

```bash
# Clone and install
git clone https://github.com/SanskarSontakke/SolarSentinel.git
cd SolarSentinel
pip install kaggle-environments numpy

# Run local benchmark (30 games vs. starter opponent)
python evaluate.py --games 30 --opponent nearest_sniper

# Test 4-player scenario
python evaluate.py --games 20 --4p
```

## How it works

The agent employs a multi-phase strategy:

1. **Candidate scoring** — Evaluates planets by production value and arrival time
2. **Coordinated attacks** — Splits fleets when targets require multi-planet coordination
3. **Evacuation** — Pulls ships from doomed planets to allies
4. **Rear consolidation** — Routes surplus rear-line ships to the front
5. **Finishing mode** — Switches to aggressive play when ahead
6. **Comet hunting** — Attempts to capture high-value comets

Key improvements over the starter agent:
- Value-scored planet targeting (not just nearest)
- Predictive interception of orbiting planets
- Active comet pursuit
- Defense simulation and garrison management
- Early/Mid/Late game phase transitions
- Sun avoidance via perpendicular waypoint routing

Tunable parameters in `submission.py`:
- `DOMINATION_HI` (0.38) — Threshold to enter finishing mode
- `DOMINATION_LO` (-0.30) — Threshold to enter desperate mode
- `HORIZON` (110) — Planning horizon in turns
- `SUN_SAFETY` (1.5) — Sun avoidance buffer
- `EARLY_TURN_LIMIT` (40) — Early game cutoff

## Results / status

Working demo. Successfully submitted to Kaggle; benchmarks vs. starter agents show 85–90% win rate against simple strategies, 60–70% vs. mid-tier bots.

## License

MIT © 2026 Sanskar Sontakke
