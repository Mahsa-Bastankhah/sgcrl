"""Builds OpenSubtaskTrain-v0's ``open/train/spawn_data.pt`` cache by reusing
CloseSubtaskTrain-v0's already-validated (collision-checked, "adjacent
room") robot spawn poses, and sampling a fresh "mostly closed"
``articulation_qpos`` for each.

Background: mshab's own ``open/train/spawn_data.pt`` on disk is the stock,
same-room version (from mshab's unpatched ``gen_open_spawn_data``). The
custom "adjacent room" spawn geometry used by this repo's
``maniskill_close_subtask_train`` comes from a hand-patched
``gen_close_spawn_data`` (geodesic-vs-Euclidean filtering, iteratively
tuned over several real jobs -- see ``env_utils.py``'s close-subtask
comments and ``slurm_scripts/maniskill_close_subtask_train/
regen_spawn_data_adjacent_room.slurm``). Porting that patch to
``gen_open_spawn_data`` and re-running the ~4hr geodesic search was judged
unnecessary: every ``close`` task-plan entry at list index i has a
byte-identical ``open`` entry at the same index i (same
``build_config_name``, ``init_config_name``, ``articulation_id``,
``articulation_handle_link_idx``, ``articulation_handle_active_joint_idx``
-- only the subtask ``uid`` differs, ``set_table-close-train-N-0`` vs.
``set_table-open-train-N-0``), so the already-validated
``robot_pos``/``robot_qpos`` for a close uid can be reused verbatim for its
matching open uid. Only ``articulation_qpos`` (the drawer's starting joint
position) needs to change, from "mostly open" (close's own
``min_open_qpos_frac=0.9`` band, i.e. ``[0.9*qmax, qmax]``) to "mostly
closed" (``[qmin, qmin + 0.1*(qmax-qmin)]`` here, the same shape of band
inverted).

For each of the 63 unique scenes in the open kitchen_counter task plan,
this builds the scene once (mirroring ``mshab/utils/gen/
gen_spawn_positions.py``'s ``make_env`` + per-scene
``env.reset(reconfigure=True, ...)`` pattern) and reads each subtask's real
``qlimits`` off the live articulation -- ``qmin`` is expected to be ~0 for
these prismatic kitchen_counter drawer joints (confirmed empirically on
close's own spawn data), but is read for real here rather than assumed.

Must run inside the ``mshab_rl`` conda env (needs ``mani_skill`` +
``mshab``); see ``build_open_spawn_data.slurm`` for the full environment
setup this depends on (same as
``slurm_scripts/maniskill_close_subtask_train/close_subtask_train.slurm``).

Reads ``MS_ASSET_DIR``/``MSHAB_TASK``/``MSHAB_SPLIT``/``MSHAB_OBJ`` env
vars, same defaults as ``env_utils.py``'s ``_mshab_close_subtask_paths``/
``_mshab_open_subtask_paths``.
"""
import os
import shutil

import gymnasium as gym
import torch
from tqdm import tqdm

import mani_skill.envs  # noqa: F401  registers 'SceneManipulation-v1'
from mani_skill.utils.scene_builder.replicacad.rearrange import (
    ReplicaCADPrepareGroceriesTrainSceneBuilder,
    ReplicaCADPrepareGroceriesValSceneBuilder,
    ReplicaCADSetTableTrainSceneBuilder,
    ReplicaCADSetTableValSceneBuilder,
    ReplicaCADTidyHouseTrainSceneBuilder,
    ReplicaCADTidyHouseValSceneBuilder,
)

from mshab.envs.planner import OpenSubtask, plan_data_from_file

# Mirrors gen_spawn_positions.py::gen_spawn_data's own dataset-name ->
# scene_builder_cls mapping.
_SCENE_BUILDER_CLS = {
    'ReplicaCADTidyHouseTrain': ReplicaCADTidyHouseTrainSceneBuilder,
    'ReplicaCADTidyHouseVal': ReplicaCADTidyHouseValSceneBuilder,
    'ReplicaCADPrepareGroceriesTrain': ReplicaCADPrepareGroceriesTrainSceneBuilder,
    'ReplicaCADPrepareGroceriesVal': ReplicaCADPrepareGroceriesValSceneBuilder,
    'ReplicaCADSetTableTrain': ReplicaCADSetTableTrainSceneBuilder,
    'ReplicaCADSetTableVal': ReplicaCADSetTableValSceneBuilder,
}

# "Mostly closed" band, the inverse of close's own "mostly open"
# min_open_qpos_frac=0.9 for kitchen_counter (gen_spawn_positions.py::
# gen_close_spawn_data): sample the bottom 10% of [qmin, qmax] instead of
# the top 10%.
_MAX_CLOSED_QPOS_FRAC = 0.1


def make_env(scene_builder_cls):
  """Mirrors gen_spawn_positions.py::make_env exactly (CPU-only sim -- this
  is a kinematics/qlimits read, not a training rollout, so no GPU is
  needed)."""
  return gym.make(
      'SceneManipulation-v1',
      num_envs=1,
      sim_backend='cpu',
      render_backend='cpu',
      reconfiguration_freq=0,
      scene_builder_cls=scene_builder_cls,
      obs_mode='state',
      reward_mode='normalized_dense',
      control_mode='pd_joint_delta_pos',
      render_mode='rgb_array',
      shader_dir='minimal',
      robot_uids='fetch',
      max_episode_steps=100,
  )


def main():
  ms_asset_dir = os.environ.get(
      'MS_ASSET_DIR', os.path.expanduser('~/.maniskill/data'))
  rearrange_dir = os.path.join(
      ms_asset_dir, 'data', 'scene_datasets', 'replica_cad_dataset',
      'rearrange')
  task = os.environ.get('MSHAB_TASK', 'set_table')
  split = os.environ.get('MSHAB_SPLIT', 'train')
  obj = os.environ.get('MSHAB_OBJ', 'kitchen_counter')

  close_spawn_fp = os.path.join(
      rearrange_dir, 'spawn_data', task, 'close', split, 'spawn_data.pt')
  open_task_plan_fp = os.path.join(
      rearrange_dir, 'task_plans', task, 'open', split, f'{obj}.json')
  open_spawn_dir = os.path.join(
      rearrange_dir, 'spawn_data', task, 'open', split)
  open_spawn_fp = os.path.join(open_spawn_dir, 'spawn_data.pt')

  print(f'Loading close spawn data from {close_spawn_fp}')
  close_spawn_data = torch.load(
      close_spawn_fp, map_location='cpu', weights_only=False)
  print(f'  {len(close_spawn_data)} close uids loaded')

  print(f'Loading open task plans from {open_task_plan_fp}')
  plan_data = plan_data_from_file(open_task_plan_fp)
  scene_builder_cls = _SCENE_BUILDER_CLS[plan_data.dataset]

  # Group task plans by scene (build_config_name) so each scene is only
  # reconfigured once -- mirrors gen_open_spawn_data's per-scene loop
  # structure (gen_spawn_positions.py).
  plans_by_scene = {}
  for tp in plan_data.plans:
    assert len(tp.subtasks) == 1 and isinstance(tp.subtasks[0], OpenSubtask), (
        f'expected a single OpenSubtask per plan, got {tp.subtasks}')
    plans_by_scene.setdefault(tp.build_config_name, []).append(tp)
  print(f'  {len(plan_data.plans)} open plans across {len(plans_by_scene)} '
        f'scenes')

  env = make_env(scene_builder_cls)
  scene_builder = env.unwrapped.scene_builder
  build_config_names_to_idxs = scene_builder.build_config_names_to_idxs
  init_config_names_to_idxs = scene_builder.init_config_names_to_idxs

  open_spawn_data = {}
  missing_close_uids = []
  for build_config_name, tps in tqdm(
      sorted(plans_by_scene.items()), desc='scenes'):
    env.reset(
        seed=0,
        options=dict(
            reconfigure=True,
            build_config_idxs=build_config_names_to_idxs[build_config_name],
        ),
    )
    for tp in tps:
      env.reset(
          seed=0,
          options=dict(
              reconfigure=False,
              init_config_idxs=init_config_names_to_idxs[tp.init_config_name],
          ),
      )
      subtask: OpenSubtask = tp.subtasks[0]
      subtask_articulation = scene_builder.articulations[
          f'env-0_{subtask.articulation_id}']
      joint_idx = subtask.articulation_handle_active_joint_idx
      qmax = subtask_articulation.qlimits[:, joint_idx, 1]
      qmin = subtask_articulation.qlimits[:, joint_idx, 0]

      open_uid = subtask.uid
      close_uid = open_uid.replace('-open-', '-close-')
      if close_uid not in close_spawn_data:
        missing_close_uids.append(open_uid)
        continue
      close_entry = close_spawn_data[close_uid]

      articulation_qpos = subtask_articulation.qpos.clone().cpu() * 0  # (1, max_dof)
      qrange = qmax - qmin
      max_closed_qpos = qrange * _MAX_CLOSED_QPOS_FRAC + qmin
      rand_qpos = torch.rand_like(qmax) * (max_closed_qpos - qmin) + qmin
      articulation_qpos[:, joint_idx] = rand_qpos.cpu()

      open_spawn_data[open_uid] = dict(
          robot_pos=close_entry['robot_pos'].clone(),
          robot_qpos=close_entry['robot_qpos'].clone(),
          articulation_qpos=articulation_qpos,
      )

  env.close()

  if missing_close_uids:
    print(f'WARNING: {len(missing_close_uids)} open uids had no matching '
          f'close uid, skipped (first 10): {missing_close_uids[:10]}')

  print(f'Built {len(open_spawn_data)} open spawn-data entries '
        f'(task plan has {len(plan_data.plans)})')

  os.makedirs(open_spawn_dir, exist_ok=True)
  if os.path.exists(open_spawn_fp):
    backup_fp = open_spawn_fp + '.orig_stock_same_room_backup'
    if not os.path.exists(backup_fp):
      print(f'Backing up existing (stock, same-room) spawn_data.pt to '
            f'{backup_fp}')
      shutil.copy(open_spawn_fp, backup_fp)
    else:
      print(f'Backup already exists at {backup_fp}, not overwriting it')

  print(f'Writing {open_spawn_fp}')
  torch.save(open_spawn_data, open_spawn_fp)
  print('Done.')


if __name__ == '__main__':
  main()
