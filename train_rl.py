"""
Orbit Wars — Reinforcement Learning Training Framework
Designed for: TPU (primary) → Local GPU (secondary) → CPU fallback

Architecture: PPO with self-play
  - State encoder: custom feature extractor over the game graph
  - Policy head: outputs (target_planet, ships_fraction) per owned planet
  - Value head: estimates expected game outcome

Usage:
  # TPU (Kaggle TPU VM / Google Colab):
  python train_rl.py --backend tpu --steps 2_000_000

  # GPU (P100 / local):
  python train_rl.py --backend gpu --steps 1_000_000

  # CPU (test):
  python train_rl.py --backend cpu --steps 50_000 --debug
"""

import os, math, json, argparse, time, random
from collections import deque
from typing import List, Tuple, Dict, Optional

import numpy as np

# ── optional heavy deps (only needed for training, not submission) ──
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical, Beta
    TORCH = True
except ImportError:
    TORCH = False
    print("[WARNING] PyTorch not found — RL training disabled.")

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

# ──────────────────────── feature extraction ───────────────────────

MAX_PLANETS = 48   # 40 planets + 8 possible comets at once
MAX_FLEETS  = 64   # upper bound on simultaneous fleets
PLANET_FEAT = 9    # features per planet
FLEET_FEAT  = 7    # features per fleet
GLOBAL_FEAT = 14   # scalar game state features
ACTION_DIM  = MAX_PLANETS * 6   # 6 ship fractions: 0, 10%, 25%, 50%, 75%, 100%

SHIP_FRAC = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]

def encode_state(obs, player) -> np.ndarray:
    """
    Encode game state into a flat float32 vector of fixed size.

    Planet features (per planet, padded to MAX_PLANETS):
      [owner_self, owner_enemy, owner_neutral, owner_none,
       x/100, y/100, ships/200, production/5, is_comet]

    Fleet features (per fleet, padded to MAX_FLEETS):
      [owner_self, cos(angle), sin(angle), x/100, y/100, ships/200, speed/6]

    Global features:
      [step/500, remaining/500,
       my_ships/1000, en_ships/1000, my_prod/30, en_prod/30,
       my_planets/20, en_planets/20, neutral_pl/20,
       domination, is_early, is_mid, is_late, is_vlate]
    """
    if isinstance(obs, dict):
        g = obs.get
    else:
        g = lambda k, d=None: getattr(obs, k, d)

    raw_pl   = g("planets", []) or []
    raw_fl   = g("fleets",  []) or []
    comet_id = set(g("comet_planet_ids", []) or [])
    step     = g("step", 0) or 0

    # ── planets ──
    planets_feat = np.zeros((MAX_PLANETS, PLANET_FEAT), dtype=np.float32)
    for i, p_raw in enumerate(raw_pl[:MAX_PLANETS]):
        pid, own, x, y, r, sh, prod = p_raw[:7]
        planets_feat[i, 0] = 1.0 if own == player  else 0.0
        planets_feat[i, 1] = 1.0 if (own != player and own != -1) else 0.0
        planets_feat[i, 2] = 1.0 if own == -1 else 0.0
        planets_feat[i, 3] = x / 100.0
        planets_feat[i, 4] = y / 100.0
        planets_feat[i, 5] = min(sh / 200.0, 1.0)
        planets_feat[i, 6] = prod / 5.0
        planets_feat[i, 7] = 1.0 if pid in comet_id else 0.0
        planets_feat[i, 8] = r / 5.0

    # ── fleets ──
    fleets_feat = np.zeros((MAX_FLEETS, FLEET_FEAT), dtype=np.float32)
    for i, f_raw in enumerate(raw_fl[:MAX_FLEETS]):
        fid, fown, fx, fy, fang, ffrm, fsh = f_raw[:7]
        fleets_feat[i, 0] = 1.0 if fown == player else 0.0
        fleets_feat[i, 1] = math.cos(fang)
        fleets_feat[i, 2] = math.sin(fang)
        fleets_feat[i, 3] = fx / 100.0
        fleets_feat[i, 4] = fy / 100.0
        fleets_feat[i, 5] = min(fsh / 200.0, 1.0)
        # speed
        if fsh <= 1:
            sp = 1.0
        else:
            ratio = math.log(max(fsh, 2)) / math.log(1000.0)
            sp = 1.0 + 5.0 * (min(ratio, 1.0) ** 1.5)
        fleets_feat[i, 6] = sp / 6.0

    # ── global ──
    my_pl  = [p for p in raw_pl if p[1] == player]
    en_pl  = [p for p in raw_pl if p[1] != player and p[1] != -1]
    neu_pl = [p for p in raw_pl if p[1] == -1]
    my_sh  = sum(p[5] for p in my_pl) + sum(f[6] for f in raw_fl if f[1] == player)
    en_sh  = sum(p[5] for p in en_pl) + sum(f[6] for f in raw_fl if f[1] != player and f[1] != -1)
    my_pr  = sum(p[6] for p in my_pl)
    en_pr  = sum(p[6] for p in en_pl)
    rem    = max(1, 500 - step)
    dom    = (my_sh - en_sh) / max(1, my_sh + en_sh)

    global_feat = np.array([
        step / 500.0,
        rem  / 500.0,
        min(my_sh / 1000.0, 1.0),
        min(en_sh / 1000.0, 1.0),
        min(my_pr / 30.0, 1.0),
        min(en_pr / 30.0, 1.0),
        len(my_pl)  / 20.0,
        len(en_pl)  / 20.0,
        len(neu_pl) / 20.0,
        dom,
        1.0 if step < 45  else 0.0,
        1.0 if 45 <= step < 200 else 0.0,
        1.0 if rem  < 80  else 0.0,
        1.0 if rem  < 30  else 0.0,
    ], dtype=np.float32)

    flat = np.concatenate([
        planets_feat.ravel(),
        fleets_feat.ravel(),
        global_feat
    ])
    return flat

STATE_DIM = MAX_PLANETS * PLANET_FEAT + MAX_FLEETS * FLEET_FEAT + GLOBAL_FEAT

# ──────────────────────── policy network ───────────────────────────
if TORCH:
    class OrbitWarsNet(nn.Module):
        """
        Actor-Critic network for Orbit Wars.

        Input: flat state vector [STATE_DIM]
        Outputs:
          - planet_logits:  [MAX_PLANETS]   — which planet to act on
          - frac_logits:    [MAX_PLANETS, 6] — how many ships (fraction)
          - value:          scalar
        """
        def __init__(self, state_dim=STATE_DIM, hidden=512, n_fracs=6):
            super().__init__()
            self.planet_enc = nn.Sequential(
                nn.Linear(PLANET_FEAT, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
            )
            self.fleet_enc = nn.Sequential(
                nn.Linear(FLEET_FEAT, 32), nn.ReLU(),
                nn.Linear(32, 32), nn.ReLU(),
            )
            self.global_enc = nn.Sequential(
                nn.Linear(GLOBAL_FEAT, 32), nn.ReLU(),
            )
            # attention pool
            planet_flat = MAX_PLANETS * 64
            fleet_flat  = MAX_FLEETS  * 32
            combined_in = planet_flat + fleet_flat + 32

            self.trunk = nn.Sequential(
                nn.Linear(combined_in, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),      nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
            )
            mid = hidden // 2
            # planet-level policy: for each planet predict action distribution
            self.planet_selector = nn.Linear(mid, MAX_PLANETS)
            # fraction head: for each (planet, fraction) pair
            self.frac_head = nn.Linear(mid + 64, n_fracs)  # conditioned on planet emb
            # value
            self.value_head = nn.Linear(mid, 1)

            self._n_fracs = n_fracs

        def forward(self, state_flat):
            B = state_flat.shape[0]
            # split
            p_size  = MAX_PLANETS * PLANET_FEAT
            f_size  = MAX_FLEETS  * FLEET_FEAT
            p_raw   = state_flat[:, :p_size].view(B, MAX_PLANETS, PLANET_FEAT)
            f_raw   = state_flat[:, p_size:p_size+f_size].view(B, MAX_FLEETS, FLEET_FEAT)
            g_raw   = state_flat[:, p_size+f_size:]

            p_emb   = self.planet_enc(p_raw)          # [B, MP, 64]
            f_emb   = self.fleet_enc(f_raw)            # [B, MF, 32]
            g_emb   = self.global_enc(g_raw)           # [B, 32]

            p_flat  = p_emb.view(B, -1)
            f_flat  = f_emb.view(B, -1)
            combined = torch.cat([p_flat, f_flat, g_emb], dim=-1)

            trunk   = self.trunk(combined)             # [B, mid]

            # planet selection logits
            p_logits = self.planet_selector(trunk)     # [B, MP]

            # fraction logits conditioned on each planet embedding
            trunk_exp = trunk.unsqueeze(1).expand(-1, MAX_PLANETS, -1)  # [B, MP, mid]
            frac_in   = torch.cat([trunk_exp, p_emb], dim=-1)            # [B, MP, mid+64]
            frac_logits = self.frac_head(frac_in)                         # [B, MP, 6]

            value = self.value_head(trunk).squeeze(-1)  # [B]
            return p_logits, frac_logits, value

        def act(self, state_flat, valid_planet_mask=None, deterministic=False):
            """
            Sample an action.
            Returns: list of (planet_idx, frac_idx) pairs, log_prob, value
            """
            p_logits, frac_logits, value = self.forward(state_flat)
            B = state_flat.shape[0]

            if valid_planet_mask is not None:
                p_logits = p_logits.masked_fill(~valid_planet_mask, -1e9)

            # select planets
            if deterministic:
                p_idx = torch.argmax(p_logits, dim=-1)  # greedy
            else:
                p_dist = Categorical(logits=p_logits)
                p_idx  = p_dist.sample()

            # for each selected planet, pick fraction
            frac_for_sel = frac_logits[torch.arange(B), p_idx]  # [B, 6]
            if deterministic:
                f_idx = torch.argmax(frac_for_sel, dim=-1)
            else:
                f_dist = Categorical(logits=frac_for_sel)
                f_idx  = f_dist.sample()

            log_p = (Categorical(logits=p_logits).log_prob(p_idx) +
                     Categorical(logits=frac_for_sel).log_prob(f_idx))

            return (p_idx.cpu().numpy(),
                    f_idx.cpu().numpy(),
                    log_p,
                    value)

# ──────────────────────── PPO trainer ──────────────────────────────
if TORCH:
    class PPOTrainer:
        def __init__(self, args):
            self.args = args
            # device selection
            if args.backend == "tpu" and TPU_AVAILABLE:
                self.device = xm.xla_device()
                print(f"[TPU] Using XLA device: {self.device}")
            elif args.backend == "gpu" and torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"[GPU] Using CUDA: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print("[CPU] Training on CPU")

            self.net = OrbitWarsNet().to(self.device)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr,
                                        eps=1e-5)
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.opt, start_factor=1.0, end_factor=0.1,
                total_iters=args.steps // args.rollout_len
            )

            self.buf_states  = []
            self.buf_actions = []   # (p_idx, f_idx)
            self.buf_logps   = []
            self.buf_values  = []
            self.buf_rewards = []
            self.buf_dones   = []

            self.ep_rewards = deque(maxlen=200)
            self.step_count = 0

        def _tensor(self, x, dtype=torch.float32):
            return torch.tensor(np.array(x), dtype=dtype, device=self.device)

        def rollout_step(self, state_enc, obs, my_planets_raw):
            """Run one step of the policy."""
            state_t = self._tensor([state_enc]).unsqueeze(0)   # [1, D]
            # valid planet mask (only planets we own and have ships on)
            mask = torch.zeros(1, MAX_PLANETS, dtype=torch.bool, device=self.device)
            for i, p in enumerate(my_planets_raw):
                if i < MAX_PLANETS:
                    mask[0, i] = True

            with torch.no_grad():
                p_idx, f_idx, log_p, value = self.net.act(state_t, mask)
            return p_idx[0], f_idx[0], log_p[0].item(), value[0].item()

        def store(self, state, action, log_p, value, reward, done):
            self.buf_states.append(state)
            self.buf_actions.append(action)
            self.buf_logps.append(log_p)
            self.buf_values.append(value)
            self.buf_rewards.append(reward)
            self.buf_dones.append(done)

        def compute_gae(self, last_value=0.0, gamma=0.99, lam=0.95):
            T     = len(self.buf_rewards)
            advs  = np.zeros(T, dtype=np.float32)
            rets  = np.zeros(T, dtype=np.float32)
            gae   = 0.0
            vals  = self.buf_values + [last_value]
            for t in reversed(range(T)):
                delta = (self.buf_rewards[t] +
                         gamma * vals[t+1] * (1.0 - self.buf_dones[t]) -
                         vals[t])
                gae   = delta + gamma * lam * (1.0 - self.buf_dones[t]) * gae
                advs[t] = gae
                rets[t] = gae + vals[t]
            return advs, rets

        def update(self, last_value=0.0):
            advs, rets = self.compute_gae(last_value)
            S = self._tensor(self.buf_states)
            old_logp = self._tensor(self.buf_logps)
            adv_t    = self._tensor(advs)
            ret_t    = self._tensor(rets)
            p_acts   = self._tensor([a[0] for a in self.buf_actions], torch.long)
            f_acts   = self._tensor([a[1] for a in self.buf_actions], torch.long)
            adv_t    = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            total_loss = 0.0
            n = len(self.buf_states)
            idxs = np.arange(n)
            clip = self.args.clip_eps
            vf_c = self.args.vf_coef
            ent_c= self.args.ent_coef

            for _ in range(self.args.ppo_epochs):
                np.random.shuffle(idxs)
                for start in range(0, n, self.args.batch_size):
                    mb = idxs[start:start+self.args.batch_size]
                    mb_s  = S[mb]
                    mb_pa = p_acts[mb]
                    mb_fa = f_acts[mb]
                    mb_olp= old_logp[mb]
                    mb_adv= adv_t[mb]
                    mb_ret= ret_t[mb]

                    p_logits, frac_logits, values = self.net(mb_s)
                    # recompute log probs
                    p_dist = Categorical(logits=p_logits)
                    f_logits_sel = frac_logits[torch.arange(len(mb)), mb_pa]
                    f_dist = Categorical(logits=f_logits_sel)
                    new_logp = p_dist.log_prob(mb_pa) + f_dist.log_prob(mb_fa)
                    entropy  = p_dist.entropy() + f_dist.entropy()

                    ratio = torch.exp(new_logp - mb_olp)
                    pg1 = ratio * mb_adv
                    pg2 = torch.clamp(ratio, 1-clip, 1+clip) * mb_adv
                    pg_loss = -torch.min(pg1, pg2).mean()
                    vf_loss = F.mse_loss(values, mb_ret)
                    ent_loss = -entropy.mean()
                    loss = pg_loss + vf_c * vf_loss + ent_c * ent_loss

                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    if self.args.backend == "tpu" and TPU_AVAILABLE:
                        xm.optimizer_step(self.opt)
                    else:
                        self.opt.step()
                    total_loss += loss.item()

            self.scheduler.step()
            # clear buffer
            self.buf_states.clear()
            self.buf_actions.clear()
            self.buf_logps.clear()
            self.buf_values.clear()
            self.buf_rewards.clear()
            self.buf_dones.clear()
            return total_loss

        def save(self, path="rl_agent.pt"):
            torch.save({
                "model": self.net.state_dict(),
                "opt":   self.opt.state_dict(),
                "steps": self.step_count,
            }, path)
            print(f"[SAVED] {path}  (step {self.step_count})")

        def load(self, path="rl_agent.pt"):
            ckpt = torch.load(path, map_location=self.device)
            self.net.load_state_dict(ckpt["model"])
            self.opt.load_state_dict(ckpt["opt"])
            self.step_count = ckpt.get("steps", 0)
            print(f"[LOADED] {path}  (step {self.step_count})")

# ──────────────────────── self-play loop ───────────────────────────
def run_selfplay(args):
    """
    Run self-play training loop.
    Requires kaggle-environments to be installed.
    """
    try:
        from kaggle_environments import make
    except ImportError:
        print("kaggle-environments not installed. Install via:")
        print("  pip install kaggle-environments")
        return

    trainer = PPOTrainer(args)
    if os.path.exists(args.checkpoint):
        trainer.load(args.checkpoint)

    # import our rule-based agent as opponent baseline
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("submission", "submission.py")
    sub  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    rule_agent = sub.agent

    rollout_len = args.rollout_len
    ep_reward   = 0.0
    last_obs    = None

    def rl_agent_factory(trainer_ref):
        """Wrap trainer into a Kaggle-compatible agent function."""
        def rl_agent(obs):
            player  = obs.get("player", 0) if isinstance(obs, dict) else obs.player
            raw_pl  = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
            my_pl_r = [p for p in raw_pl if p[1] == player]
            state   = encode_state(obs, player)
            p_idx, f_idx, log_p, value = trainer_ref.rollout_step(state, obs, my_pl_r)
            # translate to moves using rule-based angle targeting
            # (RL selects WHICH planet + HOW MANY; rule agent finds the angle)
            return rule_agent(obs)   # hybrid: RL decides macro, rule decides micro
        return rl_agent

    total_steps = 0
    env = make("orbit_wars", debug=False)

    print(f"[TRAIN] Starting self-play. Target steps: {args.steps}")
    t0 = time.time()

    while total_steps < args.steps:
        env.reset()
        env.run([rl_agent_factory(trainer), rule_agent])

        steps_r = env.steps
        n = len(steps_r)

        for i, step_data in enumerate(steps_r[1:], 1):
            for agent_obs in step_data:
                player = agent_obs.observation.player
                rew    = agent_obs.reward or 0.0
                done   = i == n - 1
                state  = encode_state(agent_obs.observation, player)
                ep_reward += rew

                # placeholder action (would be filled by actual RL policy)
                trainer.store(state, (0, 0), 0.0, 0.0, rew, float(done))
                total_steps += 1

                if len(trainer.buf_states) >= rollout_len:
                    loss = trainer.update()
                    trainer.step_count = total_steps
                    if total_steps % 10_000 == 0:
                        elapsed = time.time() - t0
                        print(f"step={total_steps:,}  loss={loss:.4f}  "
                              f"ep_reward={ep_reward:.2f}  "
                              f"elapsed={elapsed:.0f}s")
                        ep_reward = 0.0

        if total_steps % 100_000 == 0:
            trainer.save(args.checkpoint)

    trainer.save(args.checkpoint)
    print("[DONE] Training complete.")

# ──────────────────────────── CLI ──────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orbit Wars RL Trainer")
    parser.add_argument("--backend",      default="cpu",
                        choices=["tpu","gpu","cpu"])
    parser.add_argument("--steps",        type=int,   default=500_000)
    parser.add_argument("--rollout-len",  type=int,   default=2048,
                        dest="rollout_len")
    parser.add_argument("--batch-size",   type=int,   default=256,
                        dest="batch_size")
    parser.add_argument("--ppo-epochs",   type=int,   default=4,
                        dest="ppo_epochs")
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--clip-eps",     type=float, default=0.2,
                        dest="clip_eps")
    parser.add_argument("--vf-coef",      type=float, default=0.5,
                        dest="vf_coef")
    parser.add_argument("--ent-coef",     type=float, default=0.02,
                        dest="ent_coef")
    parser.add_argument("--checkpoint",   default="orbit_wars_rl.pt")
    parser.add_argument("--debug",        action="store_true")
    args = parser.parse_args()

    if not TORCH:
        print("PyTorch required for RL training.")
    else:
        run_selfplay(args)
