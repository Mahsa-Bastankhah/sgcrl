# wandb_visit_entropy_observer.py
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
    """Logs coverage/entropy & episodic return to Weights-&-Biases.
       Works for 2D or 3D obs. In 3D, z is binned into 11 layers [0..10].
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

        self._cell_size   = float(cell_size)
        self._log_every   = int(log_every_steps)
        self._rollout_len = int(rollout_len)

        self._z_levels    = 11
        self._z_min, self._z_max = 0.0, 10.0

        self._episode_counts = collections.Counter()
        self._roll_counts    = collections.Counter()
        self._ever           = set()
        self._episode_return = 0.0
        self._num_free_cells = None
        self._actor_steps    = 0
        self._is_3d          = False

    # ---------------- helpers ----------------
    def _z_to_bin(self, z: float) -> int:
        z = min(max(z, self._z_min), self._z_max)
        return int(z)

    def _cell_from_obs(self, obs) -> Tuple[int, ...]:
        v = np.asarray(obs).reshape(-1)
        # agent state is first 3 dims; goal is last 3 dims
        if v.size < 2:
            raise ValueError(f"obs too short: {v}")
        x = int(v[0] // self._cell_size)
        y = int(v[1] // self._cell_size)
        if self._is_3d:
            zbin = self._z_to_bin(float(v[2]))
            return (x, y, zbin)
        else:
            return (x, y)

    def _update(self, obs):
        cell = self._cell_from_obs(obs)
        self._episode_counts[cell] += 1
        self._roll_counts[cell]   += 1
        self._ever.add(cell)

    # ---------------- hooks ----------------
    def observe_first(self, env, timestep):
        obs = np.asarray(timestep.observation).reshape(-1)
        self._is_3d = (obs.size >= 6)  # 6 means [x,y,z,goalx,goaly,goalz]
        #print(f"[VisitEntropyObserver] obs.shape={obs.shape}, is_3d={self._is_3d}")

        if self._num_free_cells is None and hasattr(env, "walls"):
            free_2d = int((env.walls == 0).sum())
            self._num_free_cells = free_2d * (self._z_levels if self._is_3d else 1)
            #print(f"[VisitEntropyObserver] free_2d={free_2d}, "
            #      f"num_free_cells={self._num_free_cells}")

        self._episode_counts.clear()
        self._episode_return = 0.0
        self._update(timestep.observation)

    def observe(self, env, timestep, action, **unused_kwargs):
        self._actor_steps   += 1
        self._episode_return += float(timestep.reward or 0.0)
        self._update(timestep.observation)

        if self._actor_steps % self._rollout_len == 0:
            unique_roll = len(self._roll_counts)
            denom = self._num_free_cells if self._num_free_cells else max(1, len(self._ever))
            cov_roll = unique_roll / denom
            #print(f"[VisitEntropyObserver] rollout: unique={unique_roll}, denom={denom}")

            if unique_roll:
                cts   = np.fromiter(self._roll_counts.values(), dtype=np.int64)
                H     = _entropy_from_counts(cts)
                H_max = math.log(max(1, self._num_free_cells or cts.sum()))
                H_norm = H / H_max if H_max > 0 else 0.0
            else:
                H = H_norm = 0.0

            wandb.log({
                "exploration/unique_cells_rollout": float(unique_roll),
                "exploration/coverage_rollout":    float(cov_roll),
                "exploration/cell_entropy_rollout":      float(H),
                "exploration/cell_entropy_norm_rollout": float(H_norm),
            }, step=self._actor_steps)

            self._roll_counts.clear()

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
        #print(f"[VisitEntropyObserver] episode end: unique_tot={unique_tot}, denom={denom}")

        metrics = {
            "exploration/unique_cells_total":  float(unique_tot),
            "exploration/coverage_total":      float(cov_tot),
            "charts/episodic_return":          float(self._episode_return),
            "charts/global_step":              float(self._actor_steps),
            "exploration/unique_cells_episode":      float(unique_ep),
            "exploration/coverage_episode":          float(cov_ep),
            "exploration/cell_entropy_episode":      float(H_ep),
            "exploration/cell_entropy_norm_episode": float(Hn_ep),
        }

        if self._actor_steps % self._log_every == 0:
            wandb.log(metrics, step=self._actor_steps)

        return {}
