# wandb_visit_entropy_observer.py
import collections, math, wandb
from typing import Dict, Tuple
import numpy as np
from acme.utils.observers import base as observers_base

_EPS = 1e-12       # numerical safety ─ same as CleanRL


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:                      # never divide by 0
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p + _EPS)).sum())


class VisitEntropyObserver(observers_base.EnvLoopObserver):
    """Logs coverage/entropy *and* episodic return straight to Weights-&-Biases."""

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        config,                       # your whole config object (for run-name)
        cell_size: float = 1.0,
        log_every_steps: int = 1000,
        rollout_len: int = 2048,      # ← CleanRL roll-out length
    ):
        super().__init__()
        config.env_id = config.env_name.removeprefix("point_")
        # one WandB run for the whole process (safe if called several times)
        wandb.init(
            project="cleanRL",
            name=f"{config.alg_name}_{config.env_name}_{config.seed}",
            config=config.__dict__,
            reinit=False,           # do *not* create a second run if one exists
        )

        self._cell_size  = float(cell_size)
        self._log_every  = int(log_every_steps)
        self._rollout_len = int(rollout_len)            # NEW

        # running book-keeping -------------------------------------------------
        self._episode_counts = collections.Counter()    # visits in this episode
        self._roll_counts    = collections.Counter()    # visits in 2 048-step window  NEW
        self._ever           = set()                    # all-time visits
        self._episode_return = 0.0
        self._num_free_cells = None                     # filled on first reset
        self._actor_steps    = 0                        # global env-step counter

    # ---------------------------------------------------------------- helpers
    def _cell_from_obs(self, obs) -> Tuple[int, int]:
        x, y = obs[:2]
        return int(x // self._cell_size), int(y // self._cell_size)

    def _update(self, obs):
        cell = self._cell_from_obs(obs)
        self._episode_counts[cell] += 1
        self._roll_counts[cell]   += 1           # NEW
        self._ever.add(cell)

    # ------------------------------------------------------------ Acme hooks
    def observe_first(self, env, timestep):
        # Cache maze size if the env gives us a `walls` grid
        if self._num_free_cells is None and hasattr(env, "walls"):
            self._num_free_cells = int((env.walls == 0).sum())

        # reset per-episode state
        self._episode_counts.clear()
        self._episode_return = 0.0
        self._update(timestep.observation)

    def observe(self, env, timestep, action, **unused_kwargs):
        self._actor_steps   += 1
        self._episode_return += float(timestep.reward or 0.0)
        self._update(timestep.observation)

        # ---------- fixed-length CleanRL roll-out logging ----------
        if self._actor_steps % self._rollout_len == 0:
            unique_roll = len(self._roll_counts)
            cov_roll = (
                unique_roll / self._num_free_cells
                if self._num_free_cells else float("nan")
            )
            if unique_roll:
                cts   = np.fromiter(self._roll_counts.values(), dtype=np.int64)
                H     = _entropy_from_counts(cts)
                H_max = math.log(max(1, self._num_free_cells or cts.sum()))
                H_norm = H / H_max if H_max > 0 else 0.0
            else:
                H = H_norm = 0.0

            wandb.log({
                "exploration/unique_cells_rollout":      float(unique_roll),
                "exploration/coverage_rollout":          float(cov_roll),
                "exploration/cell_entropy_rollout":      float(H),
                "exploration/cell_entropy_norm_rollout": float(H_norm),
            }, step=self._actor_steps)

            self._roll_counts.clear()   # reset the 2 048-step window

    def get_metrics(self) -> Dict[str, float]:
        """Called by Acme **once per episode** – perfect place to log episode stats."""
        # ---------- per-episode stats ----------
        unique_ep = len(self._episode_counts)
        cov_ep = (
            unique_ep / self._num_free_cells
            if self._num_free_cells else float("nan")
        )
        if unique_ep:
            counts = np.fromiter(self._episode_counts.values(), dtype=np.int64)
            H_ep   = _entropy_from_counts(counts)
            H_max  = math.log(max(1, self._num_free_cells or counts.sum()))
            Hn_ep  = H_ep / H_max if H_max > 0 else 0.0
        else:
            H_ep = Hn_ep = 0.0

        # ---------- cumulative stats ----------
        unique_tot = len(self._ever)
        cov_tot = (
            unique_tot / self._num_free_cells
            if self._num_free_cells else float("nan")
        )

        metrics = {
            "exploration/unique_cells_total":  float(unique_tot),
            "exploration/coverage_total":      float(cov_tot),
            "charts/episodic_return":          float(self._episode_return),
            "charts/global_step":              float(self._actor_steps),
            # per-episode coverage/entropy (optional)
            "exploration/unique_cells_episode":      float(unique_ep),
            "exploration/coverage_episode":          float(cov_ep),
            "exploration/cell_entropy_episode":      float(H_ep),
            "exploration/cell_entropy_norm_episode": float(Hn_ep),
        }

        # log episode-level numbers occasionally
        if self._actor_steps % self._log_every == 0:
            wandb.log(metrics, step=self._actor_steps)

        # We don’t want these numbers copied into Acme CSV / TB loggers.
        return {}