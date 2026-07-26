"""Observation Normalizer and Preprocessing for BuilderBench (sgcrl).

Supports:
  - 'none': Raw observations (default).
  - 'z_scale': Static scaling of z-coordinates (z_scaled = z * z_scale_multiplier).
  - 'tied_rsnorm': Tied per-dimension Running Statistics Normalization across
    matching state and goal spatial dimensions.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


class BuilderBenchObsNormalizer:
  """Observation Preprocessor & Normalizer for BuilderBench."""

  def __init__(
      self,
      mode: str = 'none',
      num_cubes: int = 1,
      state_obs_dim: int = 4,
      goal_dim: int = 3,
      z_scale_multiplier: float = 3.0,
      rsnorm_clip: float = 10.0,
      eps: float = 1e-8,
  ):
    self.mode = str(mode).lower()
    self.num_cubes = int(num_cubes)
    self.state_obs_dim = int(state_obs_dim)
    self.goal_dim = int(goal_dim)
    self.total_obs_dim = self.state_obs_dim + self.goal_dim
    self.z_scale_multiplier = float(z_scale_multiplier)
    self.rsnorm_clip = float(rsnorm_clip)
    self.eps = float(eps)

    # Spatial position indices for state and goal (3*num_cubes)
    self.state_pos_indices = [i for i in range(3 * self.num_cubes)]
    self.goal_pos_indices = [
        self.state_obs_dim + i for i in range(3 * self.num_cubes)
    ]

    # Z-coordinate specific indices (indices 2, 5, 8, ...)
    self.state_z_indices = [3 * i + 2 for i in range(self.num_cubes)]
    self.goal_z_indices = [
        self.state_obs_dim + 3 * i + 2 for i in range(self.num_cubes)
    ]

    # Running statistics for tied_rsnorm
    self.count = 0.0
    self.mean = np.zeros(self.total_obs_dim, dtype=np.float64)
    self.var = np.ones(self.total_obs_dim, dtype=np.float64)
    self._first_apply_done = False

    print(
        f"[DEBUG] [ObsNorm] Initialized: mode={self.mode!r}, "
        f"num_cubes={self.num_cubes}, state_obs_dim={self.state_obs_dim}, "
        f"goal_dim={self.goal_dim}, z_scale={self.z_scale_multiplier}"
    )

    # Runtime assertions
    assert self.mode in ('none', 'z_scale', 'tied_rsnorm'), (
        f"Unsupported obs_norm_mode: {self.mode!r}"
    )
    assert len(self.state_z_indices) == self.num_cubes, (
        f"State Z indices count mismatch: {len(self.state_z_indices)} vs {self.num_cubes}"
    )
    assert len(self.goal_z_indices) == self.num_cubes, (
        f"Goal Z indices count mismatch: {len(self.goal_z_indices)} vs {self.num_cubes}"
    )

  def _update_running_stats(self, x: np.ndarray) -> None:
    """Parallel batch update for running mean and variance (Welford's algorithm)."""
    batch_x = x.reshape(-1, self.total_obs_dim)
    batch_count = float(batch_x.shape[0])
    if batch_count == 0:
      return

    batch_mean = np.mean(batch_x, axis=0)
    batch_var = np.var(batch_x, axis=0)

    count_prev = self.count
    if self.count == 0.0:
      self.mean = batch_mean
      self.var = batch_var
      self.count = batch_count
    else:
      new_count = self.count + batch_count
      delta = batch_mean - self.mean
      self.mean = self.mean + delta * (batch_count / new_count)
      m_a = self.var * self.count
      m_b = batch_var * batch_count
      m2 = m_a + m_b + (delta**2) * (self.count * batch_count / new_count)
      self.var = m2 / new_count
      self.count = new_count

    running_std = np.sqrt(np.maximum(self.var, self.eps))
    print(
        f"[DEBUG] [ObsNorm Stats Update] count_prev={count_prev:.0f} -> "
        f"count_new={self.count:.0f} | "
        f"Running Mean (first 3 dims): {self.mean[:3].round(4)} | "
        f"Running Std (first 3 dims): {running_std[:3].round(4)} | "
        f"Z-dim Running Std (state): {running_std[self.state_z_indices].round(4)}"
    )

  def _get_tied_stats(self) -> Tuple[np.ndarray, np.ndarray]:
    """Construct tied mean and variance for matching state and goal spatial dimensions."""
    tied_mean = self.mean.copy()
    tied_var = self.var.copy()

    for i in range(self.num_cubes):
      for axis in range(3):  # x=0, y=1, z=2
        s_idx = 3 * i + axis
        g_idx = self.state_obs_dim + 3 * i + axis

        mean_tied_val = (self.mean[s_idx] + self.mean[g_idx]) / 2.0
        var_tied_val = (self.var[s_idx] + self.var[g_idx]) / 2.0

        tied_mean[s_idx] = mean_tied_val
        tied_mean[g_idx] = mean_tied_val
        tied_var[s_idx] = var_tied_val
        tied_var[g_idx] = var_tied_val

    # Assertions to ensure state and goal tied stats are identical
    for i in range(self.num_cubes):
      for axis in range(3):
        s_idx = 3 * i + axis
        g_idx = self.state_obs_dim + 3 * i + axis
        assert np.isclose(tied_mean[s_idx], tied_mean[g_idx]), (
            f"[DEBUG] [ObsNorm] Tied mean mismatch at cube {i} axis {axis}: "
            f"{tied_mean[s_idx]} != {tied_mean[g_idx]}"
        )
        assert np.isclose(tied_var[s_idx], tied_var[g_idx]), (
            f"[DEBUG] [ObsNorm] Tied var mismatch at cube {i} axis {axis}: "
            f"{tied_var[s_idx]} != {tied_var[g_idx]}"
        )

    return tied_mean, tied_var

  def normalize(self, obs: np.ndarray, update_stats: bool = True) -> np.ndarray:
    """Preprocess / normalize observation input array."""
    obs_arr = np.asarray(obs, dtype=np.float32)
    assert obs_arr.shape[-1] == self.total_obs_dim, (
        f"[DEBUG] [ObsNorm] Obs last dim mismatch: {obs_arr.shape[-1]} != {self.total_obs_dim}"
    )

    if self.mode == 'none':
      return obs_arr

    out = obs_arr.copy()

    if self.mode == 'z_scale':
      before_z_state = np.mean(obs_arr[..., self.state_z_indices])
      before_z_goal = np.mean(obs_arr[..., self.goal_z_indices])

      # Multiply state Z and goal Z by multiplier
      for s_z in self.state_z_indices:
        out[..., s_z] *= self.z_scale_multiplier
      for g_z in self.goal_z_indices:
        out[..., g_z] *= self.z_scale_multiplier

      after_z_state = np.mean(out[..., self.state_z_indices])
      after_z_goal = np.mean(out[..., self.goal_z_indices])

      print(
          f"[DEBUG] [ObsNorm z_scale x{self.z_scale_multiplier:.1f}] "
          f"BEFORE -> State Z mean: {before_z_state:.4f}, Goal Z mean: {before_z_goal:.4f} | "
          f"AFTER -> State Z mean: {after_z_state:.4f}, Goal Z mean: {after_z_goal:.4f}"
      )
      return out

    if self.mode == 'tied_rsnorm':
      if update_stats:
        self._update_running_stats(obs_arr)

      tied_mean, tied_var = self._get_tied_stats()
      std = np.sqrt(np.maximum(tied_var, self.eps))

      before_mean = np.mean(obs_arr)
      before_std = np.std(obs_arr)

      out = (out - tied_mean) / std
      out = np.clip(out, -self.rsnorm_clip, self.rsnorm_clip)

      after_mean = np.mean(out)
      after_std = np.std(out)

      print(
          f"[DEBUG] [ObsNorm tied_rsnorm] "
          f"STATISTICS -> Count: {self.count:.0f}, Tied Mean Sample (first 3): {tied_mean[:3].round(4)}, Tied Std Sample (first 3): {std[:3].round(4)} | "
          f"BEFORE -> Mean: {before_mean:.4f}, Std: {before_std:.4f}, Min: {np.min(obs_arr):.4f}, Max: {np.max(obs_arr):.4f} | "
          f"AFTER -> Mean: {after_mean:.4f}, Std: {after_std:.4f}, Min: {np.min(out):.4f}, Max: {np.max(out):.4f}"
      )
      return out

    return out


def compute_per_axis_diagnostics(
    obs: np.ndarray, num_cubes: int, state_obs_dim: int
) -> dict[str, float]:
  """Calculate per-axis MAE, standard deviation, and ratios for logging."""
  obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1, state_obs_dim + num_cubes * 3)

  mae_x, mae_y, mae_z = 0.0, 0.0, 0.0
  std_x, std_y, std_z = 0.0, 0.0, 0.0

  for i in range(num_cubes):
    s_x = obs_arr[:, 3 * i]
    s_y = obs_arr[:, 3 * i + 1]
    s_z = obs_arr[:, 3 * i + 2]

    g_x = obs_arr[:, state_obs_dim + 3 * i]
    g_y = obs_arr[:, state_obs_dim + 3 * i + 1]
    g_z = obs_arr[:, state_obs_dim + 3 * i + 2]

    mae_x += float(np.mean(np.abs(s_x - g_x)))
    mae_y += float(np.mean(np.abs(s_y - g_y)))
    mae_z += float(np.mean(np.abs(s_z - g_z)))

    std_x += float(np.std(s_x))
    std_y += float(np.std(s_y))
    std_z += float(np.std(s_z))

  mae_x /= num_cubes
  mae_y /= num_cubes
  mae_z /= num_cubes

  std_x /= num_cubes
  std_y /= num_cubes
  std_z /= num_cubes

  mae_z_ratio = mae_z / (mae_x + mae_y + 1e-8)

  return {
      'mae_x': mae_x,
      'mae_y': mae_y,
      'mae_z': mae_z,
      'std_x': std_x,
      'std_y': std_y,
      'std_z': std_z,
      'mae_z_ratio': mae_z_ratio,
  }
