# wandb_visit_entropy_observer.py (debug version with prints)
import collections, math, wandb
from typing import Dict, Tuple
import numpy as np
from acme.utils.observers import base as observers_base

_EPS = 1e-12


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p + _EPS)).sum())


class VisitEntropyObserver(observers_base.EnvLoopObserver):
    """Logs coverage/entropy & episodic success to Weights & Biases.

    Works for 2D or 3D obs. In 3D, z is binned into integer layers [z_min..z_max].
    In noisy_tv mode, ONLY the bottom-left room has z layers; elsewhere acts 2D.
    Episodic metric is binary success (any reward >= 1 during the episode).
    """

    def __init__(self, config, cell_size: float = 1.0,
                 log_every_steps: int = 1000, rollout_len: int = 2048):
        super().__init__()
        config.env_id = config.env_name.removeprefix("point_")
        wandb.init(
            project="cleanRL",
            name=f"{config.alg_name}_{config.env_name}_{config.seed}",
            config=config.__dict__,
            reinit=False,
        )
        print("WandB initialized.", flush= True)

        self._cell_size   = float(cell_size)
        self._log_every   = int(log_every_steps)
        self._rollout_len = int(rollout_len)

        self._z_levels    = 11
        self._z_min, self._z_max = 0.0, 10.0

        self._episode_counts = collections.Counter()
        self._roll_counts    = collections.Counter()
        self._ever           = set()
        self._episode_success = 0.0
        self._num_free_cells = None
        self._actor_steps    = 0
        self._is_3d          = False

        self._noisy_tv = True
        self._grid_h = None
        self._grid_w = None

        # NEW: bottom-left step counters (episode/total/rollout) and per-episode step counter
        self._bl_steps_episode = 0
        self._bl_steps_total   = 0
        self._bl_steps_roll    = 0
        self._episode_steps    = 0  # number of observe() calls within current episode

        print("noisy_tv:", self._noisy_tv)
        

    # ---------------- helpers ----------------
    def _infer_z_config_from_env(self, env):
        if hasattr(env, "_z_min") and hasattr(env, "_z_max"):
            self._z_min = float(env._z_min)
            self._z_max = float(env._z_max)
            self._z_levels = int(round(self._z_max - self._z_min)) + 1
            #print(f"[Observer] z_min={self._z_min}, z_max={self._z_max}, z_levels={self._z_levels}")

    def _z_to_bin(self, z: float) -> int:
        z = float(min(max(z, self._z_min), self._z_max))
        return int(round(z - self._z_min))

    def _in_bottom_left_xy(self, x: float, y: float) -> bool:
        if self._grid_h is None or self._grid_w is None:
            return False
        return (x >= self._grid_h / 2.0) and (y < self._grid_w / 2.0)

    def _cell_from_obs(self, obs) -> Tuple[int, ...]:
        v = np.asarray(obs).reshape(-1)
        x = int(v[0] // self._cell_size)
        y = int(v[1] // self._cell_size)
        if self._is_3d:
            if self._noisy_tv and not self._in_bottom_left_xy(v[0], v[1]):
                return (x, y)
            zbin = self._z_to_bin(float(v[2]))
            return (x, y, zbin)
        else:
            return (x, y)

    def _update(self, obs):
        cell = self._cell_from_obs(obs)
        self._episode_counts[cell] += 1
        self._roll_counts[cell]   += 1
        self._ever.add(cell)
        #print(f"[Observer] Step {self._actor_steps}: updated cell={cell}")

    # ---------------- hooks ----------------
    def observe_first(self, env, timestep):
        obs = np.asarray(timestep.observation).reshape(-1)
        self._is_3d = (obs.size >= 6)

        # self._noisy_tv = bool(getattr(env, "_noisy_tv", False))
        if hasattr(env, "walls"):
            walls = np.asarray(env.walls)
            self._grid_h, self._grid_w = walls.shape
        self._infer_z_config_from_env(env)

        if self._num_free_cells is None and hasattr(env, "walls"):
            walls = np.asarray(env.walls)
            free_mask = (walls == 0)
            free_2d_total = int(free_mask.sum())

            if self._is_3d and self._noisy_tv:
                i_idx, j_idx = np.indices(walls.shape)
                bl_mask = (i_idx >= self._grid_h / 2.0) & (j_idx < self._grid_w / 2.0)
                free_bl = int((free_mask & bl_mask).sum())
                free_out = free_2d_total - free_bl
                self._num_free_cells = free_out * 1 + free_bl * self._z_levels
                # print(f"[Observer] noisy_tv=True, free_total={free_2d_total}, "
                #       f"free_bl={free_bl}, free_out={free_out}, "
                #       f"num_free_cells={self._num_free_cells}")
            elif self._is_3d:
                self._num_free_cells = free_2d_total * self._z_levels
                # print(f"[Observer] 3D standard, free_total={free_2d_total}, "
                #       f"num_free_cells={self._num_free_cells}")
            else:
                self._num_free_cells = free_2d_total
                #print(f"[Observer] 2D, free_total={free_2d_total}")

        self._episode_counts.clear()
        self._episode_success = 0.0

        # NEW: reset per-episode counters
        self._episode_steps = 0
        self._bl_steps_episode = 0

        self._update(timestep.observation)
        #print(f"[Observer] episode start: obs={obs}, is_3d={self._is_3d}, noisy_tv={self._noisy_tv}")

    def observe(self, env, timestep, action, **unused_kwargs):
        self._actor_steps   += 1
        # NEW: count per-episode steps
        self._episode_steps += 1

        # NEW: increment BL counters if the NEW state is in bottom-left
        v = np.asarray(timestep.observation).reshape(-1)
        if self._in_bottom_left_xy(float(v[0]), float(v[1])):
            self._bl_steps_episode += 1
            self._bl_steps_total   += 1
            self._bl_steps_roll    += 1

        r = float(timestep.reward or 0.0)
        if r >= 1.0:
            self._episode_success = 1.0
            #print(f"[Observer] SUCCESS triggered at step {self._actor_steps} (reward={r})")

        self._update(timestep.observation)

        if self._actor_steps % self._rollout_len == 0:
            unique_roll = len(self._roll_counts)
            denom = self._num_free_cells if self._num_free_cells else max(1, len(self._ever))
            cov_roll = unique_roll / denom
            #print(f"[Observer] rollout end: unique={unique_roll}, coverage={cov_roll:.3f}")

            if unique_roll:
                cts   = np.fromiter(self._roll_counts.values(), dtype=np.int64)
                H     = _entropy_from_counts(cts)
                H_max = math.log(max(1, self._num_free_cells or cts.sum()))
                H_norm = H / H_max if H_max > 0 else 0.0
                #print(f"[Observer] rollout entropy={H:.3f}, norm={H_norm:.3f}")
            self._roll_counts.clear()
            # NEW: reset per-rollout BL steps
            self._bl_steps_roll = 0

    def get_metrics(self) -> Dict[str, float]:
        denom = self._num_free_cells if self._num_free_cells else max(1, len(self._ever))
        unique_ep = len(self._episode_counts)
        cov_ep = unique_ep / denom

        if unique_ep:
            counts = np.fromiter(self._episode_counts.values(), dtype=np.int64)
            H_ep   = _entropy_from_counts(counts)
            H_max  = math.log(max(1, self._num_free_cells or counts.sum()))
            Hn_ep  = H_ep / H_max if H_max > 0 else 0.0
        else:
            H_ep = Hn_ep = 0.0

        unique_tot = len(self._ever)
        cov_tot = unique_tot / denom

        # print(f"[Observer] episode end: success={self._episode_success}, "
        #       f"unique_ep={unique_ep}, coverage_ep={cov_ep:.3f}, "
        #       f"unique_tot={unique_tot}, coverage_tot={cov_tot:.3f}")

        metrics = {
            "exploration/unique_cells_total":          float(unique_tot),
            "exploration/coverage_total":              float(cov_tot),
            "charts/episodic_success":                 float(self._episode_success),
            "charts/global_step":                      float(self._actor_steps),
            "exploration/unique_cells_episode":        float(unique_ep),
            "exploration/coverage_episode":            float(cov_ep),
            "exploration/cell_entropy_episode":        float(H_ep),
            "exploration/cell_entropy_norm_episode":   float(Hn_ep),

            # NEW: bottom-left step metrics
            "exploration/steps_bottom_left_total":      float(self._bl_steps_total),
            "exploration/steps_bottom_left_episode":    float(self._bl_steps_episode),
            "exploration/fraction_bottom_left_episode": float(
                self._bl_steps_episode / max(1, self._episode_steps)
            ),
            # Optional: uncomment if you want a rollout-scope log too
            # "exploration/steps_bottom_left_rollout":     float(self._bl_steps_roll),
        }

        #if self._actor_steps % self._log_every == 0:
        wandb.log(metrics, step=self._actor_steps)

        return {}
