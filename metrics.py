import operator
import re
from typing import Dict, Any, List, Sequence, Union
import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from optax.tree_utils import tree_l2_norm

# Additional typings
Params = flax.core.FrozenDict[str, Any]
Array = Union[np.ndarray, jnp.ndarray]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]


def get_grad_norms(grads: Union[Dict, FrozenDict], prefix: str) -> dict:
    """
    Computes the global L2 gradient norm and layer-by-layer L2 gradient norms.
    """
    metrics = {}
    
    metrics[f"{prefix}/grad_norm_total"] = tree_l2_norm(grads)
    
    flat_grads = flatten_dict(grads)
    for layer_name, grad_array in flat_grads.items():
        layer_norm = jnp.linalg.norm(jnp.array(grad_array))
        metrics[f"{prefix}/grad_norm_{layer_name}"] = layer_norm
        
    return metrics


# rephrase each key
def flatten_dict(d, parent_key="", sep="_"):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) or isinstance(v, FrozenDict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def add_prefix_to_dict(d: dict, prefix: str = None, sep="/") -> dict:
    new_dict = {}
    for key, value in d.items():
        new_dict[prefix + sep + key] = value
    return new_dict


def compute_effective_lr_paper(params, grads) -> jnp.ndarray:
    """
    From https://arxiv.org/abs/2502.15280 Hyperspherical Normalization for Scalable Deep Reinforcement Learning
    """
    p_norms = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), params)
    g_norms = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), grads)

    l1_norm_of_params_layer_i = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.abs(x)), params)
    l1_norm_of_params_total = jax.tree_util.tree_reduce(operator.add, l1_norm_of_params_layer_i)

    p_weights = jax.tree_util.tree_map(lambda x: x / (l1_norm_of_params_total + 1e-9), l1_norm_of_params_layer_i)
    eff_lrs = jax.tree_util.tree_map(lambda g, p: g / (p + 1e-9), g_norms, p_norms)

    elr_tree = jax.tree_util.tree_map(lambda w, lr: w * lr, p_weights, eff_lrs)
    elr_sum = jax.tree_util.tree_reduce(operator.add, elr_tree)
    elr = jnp.sqrt(elr_sum)

    return elr


def get_grama(
    grads: Dict[str, Dict[str, List[jnp.ndarray]]], prefix: str, tau: float = 0.1
) -> dict:
    """
    Source : https://github.com/torressliu/grad-based-plasticity-metrics/blob/5c1ea6959fccd46c8b6a8ff6b3c60caec1fbd0c1/utils/ReDo.py#L352
    """
    key = "grama"
    ratios = {}
    total_activs = []

    grads = flatten_dict(grads)

    for sub_layer_name, grad in list(grads.items()):
        if sub_layer_name.endswith('_b') or sub_layer_name.endswith('_scale') or sub_layer_name.endswith('_offset') or sub_layer_name.endswith('log_std'):
            continue
        # print("DEBUG sub_layer_name ",  sub_layer_name)
        layer_name = f"{prefix}_{sub_layer_name}"
        grad = jnp.array(grad)
        # print("DEBUG get_grama: activs.shape ", grad.shape)
        # if 'Dense' in sub_layer_name or 'dense' in sub_layer_name:
        #     score = jnp.abs(grad).mean(axis=0) # 0 instad of 1 here vs the pytorch code since in pytorch linear.weight has shape (out_features,in_features)
        # elif 'LayerNorm' in sub_layer_name:
        #     score = jnp.abs(grad)
        # elif 'BatchNorm' in sub_layer_name:
        #     score = jnp.abs(grad)
        # else:
        #     raise ValueError(f"{sub_layer_name} is not supposed to be here")
        if sub_layer_name.endswith('_w'):
          score = jnp.abs(grad).mean(axis=0) 
        else:
          raise ValueError(f"Unexpected parameter key: {sub_layer_name}")


        normalized_score = score / (score.mean() + 1e-9)
        if tau > 0.0:
            layer_mask = jnp.where(normalized_score <= tau, 1, 0)
        else:
            layer_mask = jnp.where(
                jnp.isclose(normalized_score, jnp.zeros_like(normalized_score)), 1, 0
            )

        ratios[f"{prefix}/{key}_{layer_name}_{tau}"] = (
            jnp.sum(layer_mask) / layer_mask.size
        ) * 100
        total_activs.append(layer_mask.flatten())

    if total_activs:
        total_mask = jnp.concatenate(total_activs)
        ratios[f"{prefix}/{key}_total_{tau}"] = (jnp.sum(total_mask) / total_mask.size) * 100

    return ratios


def get_srank(matrix, thershold=0.01):
    matrix = jnp.array(matrix)
    if matrix.ndim > 2:
        matrix = matrix.reshape(-1, matrix.shape[-1])
    singular_vals = jnp.linalg.svd(
        matrix, full_matrices=False, compute_uv=False)
    sum_sing = jnp.sum(singular_vals)

    # accumulated_sum = 0
    # k = 0
    # for value in singular_vals:
    #      accumulated_sum += value
    #      k += 1
    #      if accumulated_sum >= sum_sing*(1-thershold):
    #          break

    cumsum_sing = jnp.cumsum(singular_vals)
    target = sum_sing * (1 - thershold)
    k = jnp.argmax(cumsum_sing >= target) + 1

    p = singular_vals / (sum_sing + 1e-9)
    ent = -jnp.sum(p * jnp.log(p + 1e-9))
    shannon_entropy_k = jnp.exp(ent)

    return k.astype(jnp.float32), shannon_entropy_k.astype(jnp.float32)


def compute_analysis_dict(
    prefix: str,
    params: FrozenDict,
    grads: FrozenDict,
    output_repr: jnp.ndarray = None,
    execution_mask: bool = True
) -> dict:
    analysis_metrics = {}

    if grads is not None and params is not None:
        if 'actor' in prefix:
            target_params = params
        elif 'sa_encoder' in prefix:
            target_params = params['sa_encoder']
        elif 'goal_encoder' in prefix:
            target_params = params['g_encoder']
        else:
            target_params = params


        analysis_metrics.update(get_grad_norms(grads, prefix=prefix))

        analysis_metrics[f"{prefix}/effective_lr_paper"] = compute_effective_lr_paper(target_params, grads)
        analysis_metrics.update(get_grama(grads, prefix=prefix, tau=0.0))
        analysis_metrics.update(get_grama(grads, prefix=prefix, tau=0.01))
        analysis_metrics.update(get_grama(grads, prefix=prefix, tau=0.1))

    if output_repr is not None:
        def compute_srank_branch():
            return get_srank(output_repr)
        def false_branch():
            return jnp.nan, jnp.nan
            
        srank_val, eff_rank = jax.lax.cond(execution_mask, compute_srank_branch, false_branch)
        analysis_metrics[f"{prefix}/embedding_srank"] = srank_val
        analysis_metrics[f"{prefix}/embedding_effective_rank"] = eff_rank


    print("DEBUG analysis_metrics.keys() ", analysis_metrics.keys())

    return analysis_metrics
