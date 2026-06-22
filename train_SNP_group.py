import util
import numpy as np

import pandas as pd
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
import math
import os
import pickle
import zarr
from typing import List, Dict, Any, Tuple, Optional


def load_positions(pkl_path: str) -> np.ndarray:
    """Load 1D SNP physical positions (bp). Must be aligned with X[:, snp_idx, :]."""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        for _, v in obj.items():
            arr = np.asarray(v)
            if arr.ndim == 1 and arr.size > 0:
                return arr.astype(np.int64)
        raise ValueError(f"{pkl_path} is dict but no 1D positions found.")

    arr = np.asarray(obj)
    if arr.ndim != 1:
        raise ValueError(f"positions must be 1D, got shape={arr.shape}")
    return arr.astype(np.int64)


def load_groups_pkl(groups_pkl_path: str, region: str = "ext") -> List[Dict[str, Any]]:
    with open(groups_pkl_path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"{groups_pkl_path} must be a dict-like pkl.")

    if "merged_blocks" not in obj:
        raise KeyError(f"{groups_pkl_path} does not contain 'merged_blocks'.")

    merged_df = obj["merged_blocks"]
    if not isinstance(merged_df, pd.DataFrame):
        merged_df = pd.DataFrame(merged_df)

    if region not in {"core", "ext"}:
        raise ValueError("region must be 'core' or 'ext'")

    if region == "core":
        start_col = "core_start_idx_full"
        end_col = "core_end_idx_full"
        size_col = "core_n_snps"
        bp_start_col = "core_start_pos"
        bp_end_col = "core_end_pos"
    else:
        start_col = "ext_start_idx_full"
        end_col = "ext_end_idx_full"
        size_col = "ext_n_snps"
        bp_start_col = "ext_start_pos"
        bp_end_col = "ext_end_pos"

    groups_global = []
    for _, row in merged_df.iterrows():
        groups_global.append({
            "node_id": int(row["group_id"]),
            "start_global": int(row[start_col]),
            "end_global": int(row[end_col]),
            "size": int(row[size_col]),
            "bp_start": int(row[bp_start_col]),
            "bp_end": int(row[bp_end_col]),
        })

    return groups_global


def build_custom_windows_from_groups(
    groups_global: List[Dict[str, Any]],
    start_idx: int,
    end_idx: int,
) -> List[Dict[str, Any]]:
    windows = []
    for g in groups_global:
        gs, ge = g["start_global"], g["end_global"]
        if ge < start_idx or gs >= end_idx:
            continue

        inter_s = max(gs, start_idx)
        inter_e = min(ge, end_idx - 1)

        windows.append({
            "window_id": int(g["node_id"]),
            "start": int(inter_s - start_idx),      # local slice coord
            "end": int(inter_e - start_idx + 1),    # half-open
            "global_start": int(inter_s),
            "global_end": int(inter_e),
            "size": int(inter_e - inter_s + 1),
            "bp_start": int(g.get("bp_start", -1)),
            "bp_end": int(g.get("bp_end", -1)),
        })
    return windows


def groups_intersecting_window(
    groups_global: List[Dict[str, Any]],
    start_idx: int,
    end_idx: int,
) -> List[Dict[str, Any]]:
    """
    Map global groups to a window slice [start_idx, end_idx) along SNP index axis.
    Keep ONLY the intersection (clipped) range inside this window and return local coords.
    """
    w_left = start_idx
    w_right = end_idx - 1
    out = []
    for g in groups_global:
        gs, ge = g["start_global"], g["end_global"]
        if ge < w_left or gs > w_right:
            continue
        inter_s = max(gs, w_left)
        inter_e = min(ge, w_right)
        out.append({
            "node_id": g["node_id"],
            "local_start": inter_s - start_idx,
            "local_end": inter_e - start_idx,
            "global_start": inter_s,
            "global_end": inter_e,
        })
    return out


def print_first_k_groups(groups: List[Dict[str, Any]], k: int = 10) -> None:
    print(f"\nShowing first {min(k, len(groups))} groups:")
    for g in groups[:k]:
        if "start_global" in g and "end_global" in g and "size" in g:
            print(
                f"- node_id={g['node_id']} | idx_global=[{g['start_global']}..{g['end_global']}] | "
                f"size={g['size']} | bp=[{g.get('bp_start','?')}..{g.get('bp_end','?')}]"
            )
        else:
            print(
                f"- node_id={g['node_id']} | local=[{g['local_start']}..{g['local_end']}] | "
                f"global=[{g['global_start']}..{g['global_end']}]"
            )


def maybe_shuffle_custom_windows(
    custom_windows: List[Dict[str, Any]],
    shuffle_group_order: bool = False,
    shuffle_seed: int = 1234,
) -> List[Dict[str, Any]]:
    """
    Keep each group/window unchanged, but randomly permute the ORDER of windows.
    This is useful for testing whether the model relies on global group order.
    """
    if not shuffle_group_order:
        return custom_windows

    rng = np.random.default_rng(shuffle_seed)
    order = np.arange(len(custom_windows))
    rng.shuffle(order)

    shuffled = [custom_windows[i] for i in order]

    print(f"[Shuffle group order] enabled=True | seed={shuffle_seed}")
    print(f"[Shuffle group order] first 10 order: {order[:10].tolist()}")

    return shuffled

def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def main(train_config, train_seed, split_seed):
    set_all_seeds(train_seed)

    x_path = train_config.get("x_zarr", f"{train_config['dataset']}/x_train.zarr")
    y_path = train_config.get("y_path", f"{train_config['dataset']}/y_train.pkl")

    shuffle_group_order = bool(train_config.get("shuffle_group_order", False))
    shuffle_group_seed = int(train_config.get("shuffle_group_seed", split_seed))

    x = zarr.open(x_path, mode="r")
    y = pd.read_pickle(y_path)
    y = np.asarray(y)

    print("[Loaded]")
    print("  x dtype/shape:", x.dtype, x.shape)
    print("  y dtype/shape:", y.dtype, y.shape)
    print("  y first 20:", np.asarray(y).reshape(-1)[:20])

    if y.ndim == 2 and y.shape[1] > 1:
        y_strat = np.argmax(y, axis=1)
    else:
        y_strat = y.reshape(-1)

    indices = np.arange(len(y))
    idx_train, idx_valid, y_train, y_valid = train_test_split(
        indices, y,
        stratify=y_strat,
        test_size=train_config["valid_size"],
        random_state=split_seed,
    )

    x_train_unsliced = x[idx_train]
    x_valid_unsliced = x[idx_valid]

    if x.ndim == 3:
        assert x.shape[2] == 2, f"Expected genotype channels=2, got {x.shape}"

    for model_config in train_config["models"]:
        params = model_config.get("params", {})
        window_mode = params.get("window_mode", "fixed")
        group_region = params.get("group_region", "ext")

        if "window_overlap" in train_config and "window_overlap" not in params:
            params["window_overlap"] = train_config["window_overlap"]

        if x.ndim == 3 and "window_size" in params:
            ws = params["window_size"]
            params.setdefault("local_model_config", {}).setdefault("params", {})["input_length"] = ws
            params["local_model_config"]["params"]["add_embedding_layer"] = False

        # slice dataset first
        if "dataset_slice" in model_config:
            dataset_slice = model_config.get("dataset_slice", None)
            if dataset_slice is None:
                start_idx, end_idx = 0, x.shape[1]
            else:
                start_idx, end_idx = dataset_slice

            x_train = util.slice_dataset(x_train_unsliced, start_idx, end_idx)
            x_valid = util.slice_dataset(x_valid_unsliced, start_idx, end_idx)

            if x_train.ndim >= 2:
                params["input_length"] = x_train.shape[1]
        else:
            start_idx, end_idx = 0, x.shape[1]
            x_train = x_train_unsliced
            x_valid = x_valid_unsliced
            if x_train.ndim >= 2:
                params["input_length"] = x_train.shape[1]

        # integrate LD-group windows if requested
        if window_mode == "ld_groups":
            groups_pkl = train_config.get("groups_pkl", None)
            if groups_pkl is None:
                raise ValueError("window_mode='ld_groups' requires train_config['groups_pkl'].")

            groups_global = load_groups_pkl(groups_pkl, region=group_region)
            print(f"[Loaded LD groups] n_groups={len(groups_global)}")
            print_first_k_groups(groups_global, k=10)

            custom_windows = build_custom_windows_from_groups(
                groups_global=groups_global,
                start_idx=start_idx,
                end_idx=end_idx,
            )
            print(f"[Custom windows from groups] n_windows={len(custom_windows)}")

            custom_windows = maybe_shuffle_custom_windows(
                custom_windows,
                shuffle_group_order=shuffle_group_order,
                shuffle_seed=shuffle_group_seed,
            )

            # pass custom windows into model
            params["custom_windows"] = custom_windows
            params["window_mode"] = "ld_groups"

            # optional compatibility for dynamic first layer / masking
            params["group_spans"] = [(w["start"], w["end"] - 1) for w in custom_windows]

        else:
            params["custom_windows"] = None
            params["window_mode"] = "fixed"

        if x_train.ndim == 3 and "window_size" in params and window_mode == "fixed":
            ws = params["window_size"]
            assert x_train.shape[1] >= ws, f"Not enough SNPs: {x_train.shape[1]} < window_size {ws}"

        model = util.build_model(model_config["type"], params)

        if model_config.get("type", "").lower() == "windowedensemble":
            epochs = model_config["params"]["global_model_config"]["params"]["epochs"]
        else:
            epochs = model_config["params"]["epochs"]

        model.train(
            x_train, y_train,
            x_valid, y_valid,
            epochs,
            model_config["save_to"],
            model_config["name"]
        )

        model.save(model_config["save_to"], model_config["name"])


if __name__ == "__main__":
    exp_config = util.get_exp_config("Train a series of models on a specified dataset.")

    if "train" in exp_config.keys():
        train_configs = exp_config["train"]

        for i, train_config in enumerate(train_configs):
            print(f"\n# -- Running Training Configuration {i+1}/{len(train_configs)} -- #\n")
            train_seed = int(train_config.get("train_seed", exp_config.get("train_seed", exp_config.get("random_seed", 512))))
            split_seed = int(train_config.get("split_seed", exp_config.get("split_seed", 1234)))
            print(f"[Seeds] train_seed={train_seed} | split_seed={split_seed}")
            main(train_config, train_seed, split_seed)
    else:
        print("No Training Configuration Provided: Skipping Stage...")