"""Checkpoint-level reward-network spectral norm, burned into probe videos.

For a frozen checkpoint this is one scalar: max_ℓ σ_max(W_ℓ) over Linear
kernels (and reconstructed InvertiblePLU maps). Overlay is a top bar so the
existing reward strip is not covered.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import pickle

import jax  # noqa: F401  — needed to unpickle jax.Array checkpoints
import numpy as np


def _as_np(x) -> np.ndarray:
  return np.asarray(x)


def _sigma_max(w: np.ndarray) -> float:
  w = np.asarray(w, dtype=np.float64)
  if w.ndim != 2 or min(w.shape) == 0:
    return float('nan')
  # Spectral norm = largest singular value.
  return float(np.linalg.svd(w, compute_uv=False)[0])


def _is_plu_dict(d: dict) -> bool:
  return isinstance(d, dict) and all(k in d for k in ('L', 'U', 's'))


def _plu_W(d: dict) -> np.ndarray:
  L_free = _as_np(d['L'])
  U_free = _as_np(d['U'])
  s = _as_np(d['s'])
  dim = int(s.shape[0])
  P = _as_np(d['P']) if 'P' in d else np.eye(dim)
  L = np.tril(L_free, k=-1) + np.eye(dim)
  U = np.triu(U_free, k=1) + np.diag(s)
  return P @ L @ U


def collect_linear_maps(tree, prefix: str = '') -> list[tuple[str, np.ndarray]]:
  """Linear maps whose σ_max we report: Haiku ``w`` kernels + PLU W."""
  out: list[tuple[str, np.ndarray]] = []
  if isinstance(tree, dict):
    if _is_plu_dict(tree):
      out.append((prefix.rstrip('/') + '/W', _plu_W(tree)))
      return out
    for k, v in tree.items():
      path = f'{prefix}{k}'
      if k in ('w', 'kernel') and hasattr(v, 'shape') and np.asarray(v).ndim == 2:
        out.append((path, _as_np(v)))
      else:
        out.extend(collect_linear_maps(v, path + '/'))
  return out


def reward_spec_norm(q_params) -> dict:
  maps = collect_linear_maps(q_params)
  if not maps:
    raise RuntimeError('no Linear/PLU weight matrices found in q_params')
  specs = [(name, _sigma_max(w), tuple(w.shape)) for name, w in maps]
  specs.sort(key=lambda t: t[1], reverse=True)
  return {
      'sigma_max': specs[0][1],
      'argmax_layer': specs[0][0],
      'argmax_shape': specs[0][2],
      'n_layers': len(specs),
      'layers': [
          {'name': n, 'sigma_max': s, 'shape': list(sh)}
          for n, s, sh in specs[:12]
      ],
  }


def _find_font() -> str:
  for p in (
      '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf',
      '/usr/share/fonts/dejavu/DejaVuSans.ttf',
      '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
      '/usr/share/fonts/liberation/LiberationSans-Bold.ttf',
  ):
    if os.path.isfile(p):
      return p
  raise FileNotFoundError('no DejaVu/Liberation font for ffmpeg drawtext')


def overlay_top_bar(mp4_in: str, mp4_out: str, label: str, *, bar_h: int = 52) -> None:
  font = _find_font()
  # Escape ffmpeg drawtext special chars.
  text = (label.replace('\\', '\\\\').replace("'", r"\'")
          .replace(':', r'\:').replace('%', r'\%'))
  vf = (
      f'pad=iw:ih+{bar_h}:0:{bar_h}:color=0x0f1419,'
      f'drawtext=fontfile={font}:text=\'{text}\':'
      f'x=16:y=14:fontsize=22:fontcolor=0xe8eef4'
  )
  tmp = mp4_out + '.tmp.mp4'
  cmd = [
      'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
      '-i', mp4_in, '-vf', vf, '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
      '-crf', '18', tmp,
  ]
  subprocess.run(cmd, check=True)
  os.replace(tmp, mp4_out)


DEFAULT_JOBS = [
    {
        'ckpt': (
            'logs/ppo_builderbench_creative7_task2_e1024_pd_crl_tau05_nopermute_'
            'fixedx01_catwp_extrew1_minstd1e5_entanneal_ep70_300m/'
            'ppo_builderbench_creative_7_task2_0/checkpoints/ckpt_iter_0000450.pkl'
        ),
        'mp4': (
            'figs/builderbench/frozen_crl_reward_probe/'
            'c7t2_crl_catwp_ms_ea_s0_online_all/'
            'c7t2_crl_catwp_ms_ea_s0_online_iter_0000450.mp4'
        ),
        'kind': 'phi·psi',
    },
    {
        'ckpt': (
            'logs/ppo_builderbench_creative7_task2_e1024_pd_crl_tau05_nopermute_'
            'fixedx01_catwp_extrew1_minstd1e5_entanneal_ep70_300m/'
            'ppo_builderbench_creative_7_task2_0/checkpoints/ckpt_iter_0000600.pkl'
        ),
        'mp4': (
            'figs/builderbench/frozen_crl_reward_probe/'
            'c7t2_crl_catwp_ms_ea_s0_online_all/'
            'c7t2_crl_catwp_ms_ea_s0_online_iter_0000600.mp4'
        ),
        'kind': 'phi·psi',
    },
    {
        'ckpt': (
            'logs/ppo_builderbench_creative7_task2_e1024_pd_crl_tau05_nopermute_'
            'fixedx01_catwp_extrew1_minstd1e5_entanneal_ep70_300m/'
            'ppo_builderbench_creative_7_task2_0/checkpoints/ckpt_iter_0000750.pkl'
        ),
        'mp4': (
            'figs/builderbench/frozen_crl_reward_probe/'
            'c7t2_crl_catwp_ms_ea_s0_online_all/'
            'c7t2_crl_catwp_ms_ea_s0_online_iter_0000750.mp4'
        ),
        'kind': 'phi·psi',
    },
    {
        'ckpt': (
            'logs/ppo_builderbench_creative5_task2_e1024_pd_nf_tau05_actorreset_'
            'evalvid_catselect_extrew1/ppo_builderbench_creative_5_task2_1/'
            'checkpoints/ckpt_iter_0000600.pkl'
        ),
        'mp4': (
            'figs/builderbench/nf_logp_reward_probe/'
            'c5t2_nf_extrew1_s1_online_all/'
            'c5t2_nf_extrew1_s1_online_iter_0000600.mp4'
        ),
        'kind': 'NF',
    },
    {
        'ckpt': (
            'logs/ppo_builderbench_creative5_task2_e1024_pd_nf_tau05_actorreset_'
            'evalvid_catselect_extrew1/ppo_builderbench_creative_5_task2_1/'
            'checkpoints/ckpt_iter_0000750.pkl'
        ),
        'mp4': (
            'figs/builderbench/nf_logp_reward_probe/'
            'c5t2_nf_extrew1_s1_online_all/'
            'c5t2_nf_extrew1_s1_online_iter_0000750.mp4'
        ),
        'kind': 'NF',
    },
    {
        'ckpt': (
            'logs/ppo_builderbench_creative5_task2_e1024_pd_nf_tau05_actorreset_'
            'evalvid_catselect_extrew1/ppo_builderbench_creative_5_task2_1/'
            'checkpoints/ckpt_iter_0003600.pkl'
        ),
        'mp4': (
            'figs/builderbench/nf_logp_reward_probe/'
            'c5t2_nf_extrew1_s1_online_all/'
            'c5t2_nf_extrew1_s1_online_iter_0003600.mp4'
        ),
        'kind': 'NF',
    },
]


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--out_json', default='')
  args = p.parse_args()
  os.chdir(_REPO)

  summary = []
  for job in DEFAULT_JOBS:
    ckpt_path = os.path.join(_REPO, job['ckpt'])
    mp4_path = os.path.join(_REPO, job['mp4'])
    print(f'[spec] load {ckpt_path}', flush=True)
    with open(ckpt_path, 'rb') as fh:
      ckpt = pickle.load(fh)
    stats = reward_spec_norm(ckpt['q_params'])
    sigma = stats['sigma_max']
    label = (
        f"reward spec.norm = {sigma:.3f}  "
        f"({job['kind']}, frozen ckpt, max_l sigma_max(W_l))"
    )
    print(f'[spec] {os.path.basename(mp4_path)}  σ_max={sigma:.6g}  '
          f'layer={stats["argmax_layer"]}  n={stats["n_layers"]}', flush=True)
    if not os.path.isfile(mp4_path):
      raise FileNotFoundError(mp4_path)
    overlay_top_bar(mp4_path, mp4_path, label)
    rec = {
        'mp4': job['mp4'],
        'ckpt': job['ckpt'],
        'kind': job['kind'],
        **stats,
        'label': label,
    }
    summary.append(rec)

  out_json = args.out_json or os.path.join(
      _REPO, 'figs/builderbench/reward_weight_spec_norm_overlay.json')
  os.makedirs(os.path.dirname(out_json), exist_ok=True)
  with open(out_json, 'w', encoding='utf-8') as fh:
    json.dump(summary, fh, indent=2)
  print(f'[spec] wrote {out_json}', flush=True)
  for rec in summary:
    print(f"  {os.path.basename(rec['mp4'])}: {rec['sigma_max']:.4f}",
          flush=True)


if __name__ == '__main__':
  main()
