
import os
import json
import pickle
import zarr
import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_snp_map(load_from: str, model_name: str) -> dict:
    """
    Legacy: load SNP selection map for models trained on a subset of SNP indices.

    If the map file is absent and use_snp_map=False in config, callers should not invoke this.
    """
    path = os.path.join(load_from, f"{model_name}_snp_map.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"SNP map not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unwrap_shap_values(shap_vals) -> np.ndarray:
    """
    SHAP may return:
      - numpy ndarray
      - list[ndarray] (multi-output)
      - shap.Explanation (new API) with .values
    This converts to a numpy ndarray of values.
    """
    if isinstance(shap_vals, list) and len(shap_vals) > 0:
        shap_vals = shap_vals[0]
    if hasattr(shap_vals, "values"):
        return np.asarray(shap_vals.values)
    return np.asarray(shap_vals)


def compute_shap(model, x_bg: np.ndarray, x_test: np.ndarray):
    """
    Compute SHAP values for a (wrapped) model.
    - If model implements get_shap_explanation(x_bg, x_test), use it.
    - Else fall back to shap.DeepExplainer(model.model, x_bg).
    """

    if hasattr(model, "get_shap_explanation"):
        out = model.get_shap_explanation(x_bg, x_test)
        print("[DEBUG] get_shap_explanation returned shape:",
        np.asarray(out.values if hasattr(out, "values") else out).shape,
        "| x_test shape:", x_test.shape)

        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    import shap
    x_bg = x_bg.astype(np.float32)
    x_test = x_test.astype(np.float32)

    if not hasattr(model, "model"):
        raise AttributeError("Model has no attribute '.model' for SHAP DeepExplainer fallback.")

    explainer = shap.DeepExplainer(model.model, x_bg)
    shap_vals = explainer.shap_values(x_test)
    return shap_vals


def compute_shap_in_chunks(model, x_bg: np.ndarray, x_test: np.ndarray, chunk_size: int = 8, desc: str = "SHAP"):
    """
    Compute SHAP values in chunks to reduce peak RAM / GPU usage.
    Returns a single concatenated numpy array over all test samples.
    """
    x_test = np.asarray(x_test)
    if x_test.shape[0] == 0:
        raise ValueError(f"{desc}: x_test is empty.")
    if chunk_size <= 0:
        raise ValueError(f"{desc}: chunk_size must be > 0, got {chunk_size}")

    parts = []
    total = x_test.shape[0]
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        print(f"[DEBUG] {desc} chunk: {start}:{end} / {total}")
        shap_part = compute_shap(model, x_bg, x_test[start:end])
        sv_part = np.asarray(_unwrap_shap_values(shap_part))
        parts.append(sv_part)

    return np.concatenate(parts, axis=0)


def plot_shap_bar_like_example(feature_names, mean_abs, out_png, top_k=10, xlabel="mean(|SHAP value|)"):
    """Horizontal SHAP-style bar plot (like your example)."""
    mean_abs = np.asarray(mean_abs, dtype=float).reshape(-1)
    k = min(int(top_k), mean_abs.size)
    if k <= 0:
        return
    idx = np.argsort(mean_abs)[::-1][:k]
    vals = mean_abs[idx][::-1]  # smallest at bottom, largest at top
    names = np.asarray(feature_names, dtype=object)[idx][::-1]

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    y = np.arange(len(vals))
    ax.barh(y, vals, color="#ff0050", alpha=0.95)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=11)

    ax.xaxis.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_max = float(max(vals.max(), 1e-12))
    for yi, v in zip(y, vals):
        ax.text(v + x_max * 0.02, yi, f"+{v:.2f}", va="center", ha="left",
                fontsize=10, color="#ff0050")
    ax.set_xlim(0, x_max * 1.15)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    

def plot_shap_heatmap_like_example(
    shap_matrix,              # (n_features, n_instances), signed SHAP
    feature_names,
    out_png,
    fx=None,                  # optional (n_instances,)
    max_display=None,         # None => show all
    colorbar_label="SHAP value (impact on model output)"
):
    """SHAP-style heatmap with right mean(|SHAP|) bars and optional top f(x) line.

    - Uses a pink/blue diverging colormap (blue=negative, pink=positive, white=0).
    - If max_display is None, displays all features (recommended for window-level global plots).
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.gridspec import GridSpec

    PINKBLUE = LinearSegmentedColormap.from_list(
        "pinkblue", ["#1f6feb", "#ffffff", "#ff2d7a"], N=256
    )

    S = np.asarray(shap_matrix, dtype=float)
    if S.ndim != 2:
        raise ValueError(f"shap_matrix must be 2D (features x instances), got {S.shape}")

    n_feat, n_inst = S.shape
    mean_abs = np.mean(np.abs(S), axis=1)

    # Optional top-k display
    if max_display is None:
        order = np.arange(n_feat)
    else:
        order = np.argsort(mean_abs)[::-1][:min(int(max_display), n_feat)]

    S = S[order, :]
    names = np.asarray(feature_names, dtype=object)[order]
    mean_abs = mean_abs[order]
    n_feat_disp = S.shape[0]

    # Dynamic sizing so y labels don't collapse
    row_h = 0.24
    fig_h = max(10.0, n_feat_disp * row_h)
    fig_w = 16.5

    y_font = 8 if n_feat_disp <= 60 else (6 if n_feat_disp <= 120 else 5)

    vmax = float(np.percentile(np.abs(S), 99))
    vmax = max(vmax, 1e-12)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    top_h = 1.2
    main_h = max(6.0, n_feat_disp * 0.35)
    gs = GridSpec(
        nrows=2, ncols=3,
        width_ratios=[20, 2.8, 0.7],
        height_ratios=[top_h, main_h],
        wspace=0.03, hspace=0.02
    )

    # top f(x)
    ax_fx = fig.add_subplot(gs[0, 0])
    fx_plot = None
    if fx is not None:
        fx_arr = np.asarray(fx, dtype=float).reshape(-1)
        if fx_arr.shape[0] == n_inst:
            fx_plot = fx_arr
    if fx_plot is not None:
        ax_fx.plot(np.arange(n_inst), fx_plot, color="black", linewidth=1.2)
        ax_fx.set_ylabel("f(x)", fontsize=10)
    ax_fx.set_xlim(0, max(0, n_inst - 1))
    ax_fx.set_xticks([])
    ax_fx.spines["top"].set_visible(False)
    ax_fx.spines["right"].set_visible(False)

    # main heatmap
    ax_hm = fig.add_subplot(gs[1, 0])
    im = ax_hm.imshow(
        S,
        aspect="auto",
        interpolation="nearest",
        cmap=PINKBLUE,
        vmin=-vmax, vmax=vmax
    )
    ax_hm.set_yticks(np.arange(n_feat_disp))
    ax_hm.set_yticklabels(names, fontsize=y_font)
    ax_hm.set_xlabel("Sample number", fontsize=10)
    num_ticks = min(n_inst, 20) # 最多显示20个刻度
    ax_hm.set_xticks(np.linspace(0, max(0, n_inst - 1), num_ticks).astype(int))

    # right bars: mean(|SHAP|)
    # 右侧 mean(|SHAP|) 黑条（不要 sharey，避免把主图 ytick 清掉）
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_bar.barh(np.arange(n_feat_disp), mean_abs, color="black", alpha=0.95)
    ax_bar.invert_xaxis()  # 贴近热图
    ax_bar.set_xticks([])

    # ✅ 对齐 y 范围到热图，但不影响热图的 yticklabels
    ax_bar.set_ylim(ax_hm.get_ylim())
    ax_bar.tick_params(left=False, labelleft=False)

    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    # colorbar
    ax_cb = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(im, cax=ax_cb)
    cbar.set_label(colorbar_label, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(values_1d: np.ndarray, xlabels: list, out_png: str, ylabel: str = "mean(|SHAP|)"):
    """Legacy 1D heatmap (kept for compatibility)."""
    heat = np.asarray(values_1d, dtype=float).reshape(1, -1)
    fig, ax = plt.subplots(figsize=(12, 1.8), dpi=150)
    ax.imshow(heat, aspect="auto", interpolation="nearest", cmap="coolwarm")
    tick_step = max(1, len(xlabels) // 12)
    ticks = list(range(0, len(xlabels), tick_step))
    labels = [str(xlabels[t]) for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks([0])
    ax.set_yticklabels([ylabel])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_bar(mean_abs, feature_index_map, out_png, top_n=None, ylabel="mean(|SHAP|)"):
    """
    Draw a SHAP-style horizontal bar plot.
    - If top_n is None, plots all features (recommended for window/global bar plots).
    - Auto-scales figure height so y labels remain readable.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    vals = np.asarray(mean_abs, dtype=float).reshape(-1)
    F = vals.shape[0]

    if feature_index_map is None:
        names = np.array([f"feat_{i}" for i in range(F)], dtype=object)
    else:
        fim = np.asarray(feature_index_map).reshape(-1)
        names = np.array([str(x) for x in fim], dtype=object)

    if top_n is None:
        top_n = F
    top_n = min(int(top_n), F)

    idx = np.argsort(vals)[::-1][:top_n]
    top_vals = vals[idx][::-1]
    top_names = names[idx][::-1]

    row_h = 0.28
    fig_h = max(6.0, top_n * row_h)
    fig_w = 9.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    y = np.arange(top_n)

    ax.barh(y, top_vals, color="#ff0050", alpha=0.95)
    ax.set_yticks(y)
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel(ylabel, fontsize=11)

    ax.xaxis.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_max = max(float(top_vals.max()), 1e-12)
    for yi, v in zip(y, top_vals):
        ax.text(v + x_max * 0.01, yi, f"+{v:.2f}", va="center", ha="left",
                fontsize=9, color="#ff0050")

    ax.set_xlim(0, x_max * 1.12)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def mean_abs_feature_from_shap(shap_vals) -> np.ndarray:
    """
    Generic mean(|SHAP|) over samples for arbitrary feature vector SHAP.
    Returns (F,).
    """
    sv = _unwrap_shap_values(shap_vals)

    # Common shapes:
    # (N, F)
    # (N, F, 1) or (N, F, C)
    # (N, T, F)  -> treat last dim as features, average over N and T
    if sv.ndim == 1:
        return np.abs(sv).reshape(-1)
    if sv.ndim == 2:
        return np.abs(sv).mean(axis=0).reshape(-1)
    if sv.ndim == 3:
        # If last dim is 1, squeeze; else average over all but last axis
        if sv.shape[-1] == 1:
            sv2 = np.squeeze(sv, axis=-1)
            print("[DEBUG] after reshape sv2 shape:", sv2.shape)
            return np.abs(sv2).mean(axis=0).reshape(-1)
        return np.abs(sv).mean(axis=tuple(range(sv.ndim - 1))).reshape(-1)
    # fallback
    return np.abs(sv).mean(axis=tuple(range(sv.ndim - 1))).reshape(-1)


def mean_abs_snp_from_shap(shap_vals, L: int) -> np.ndarray:
    """
    Convert SHAP attributions back to SNP-level importance.

    Supported common shapes:
      - (N, L)          : one attribution per SNP
      - (N, L, 2)       : 2-channel genotype attribution per SNP
      - (N, L, 1)       : single-channel 3D case
      - (N, L*2)        : flattened 2-channel genotype input
      - (N, L, 2, 1)    : occasionally returned by some wrappers

    Returns:
      - shape (L,)
    """
    sv = _unwrap_shap_values(shap_vals)
    sv = np.asarray(sv)

    print(f"[DEBUG] mean_abs_snp_from_shap input shape: {sv.shape}, L={L}")

    if sv.ndim == 2 and sv.shape[1] == L:
        out = np.mean(np.abs(sv), axis=0)
        print(f"[DEBUG] SNP importance shape from (N,L): {out.shape}")
        return out.reshape(-1)

    if sv.ndim == 2 and sv.shape[1] == L * 2:
        out = np.mean(np.abs(sv), axis=0).reshape(L, 2).mean(axis=1)
        print(f"[DEBUG] SNP importance shape from (N,L*2): {out.shape}")
        return out.reshape(-1)

    if sv.ndim == 3 and sv.shape[1] == L and sv.shape[2] == 2:
        out = np.mean(np.abs(sv), axis=(0, 2))
        print(f"[DEBUG] SNP importance shape from (N,L,2): {out.shape}")
        return out.reshape(-1)

    if sv.ndim == 3 and sv.shape[1] == L and sv.shape[2] == 1:
        out = np.mean(np.abs(sv[..., 0]), axis=0)
        print(f"[DEBUG] SNP importance shape from (N,L,1): {out.shape}")
        return out.reshape(-1)

    if sv.ndim == 4 and sv.shape[1] == L and sv.shape[2] == 2:
        out = np.mean(np.abs(sv), axis=(0, 2, 3))
        print(f"[DEBUG] SNP importance shape from (N,L,2,1): {out.shape}")
        return out.reshape(-1)

    if sv.ndim > 2:
        flat = sv.reshape(sv.shape[0], -1)
    else:
        flat = sv

    F = flat.shape[1]
    print(f"[WARN] Unexpected SHAP shape {sv.shape}; flattened to {flat.shape}")

    if F == L:
        out = np.mean(np.abs(flat), axis=0)
        return out.reshape(-1)

    if F == L * 2:
        out = np.mean(np.abs(flat), axis=0).reshape(L, 2).mean(axis=1)
        return out.reshape(-1)

    raise ValueError(
        f"Cannot convert SHAP values to SNP-level importance: "
        f"SHAP shape={sv.shape}, flattened feature width={F}, expected L={L} or 2L={L*2}"
    )


def shap_to_snp_matrix(shap_vals, L: int) -> np.ndarray:
    """
    Convert SHAP values to a per-sample x per-SNP signed matrix with shape (N, L).

    For 2-channel genotype input, SNP-level SHAP is computed by summing channel
    contributions, because SHAP values are additive attributions.
    """
    sv = np.asarray(_unwrap_shap_values(shap_vals), dtype=float)
    print(f"[DEBUG] shap_to_snp_matrix input shape: {sv.shape}, L={L}")

    if sv.ndim == 2 and sv.shape[1] == L:
        return sv

    if sv.ndim == 2 and sv.shape[1] == L * 2:
        return sv.reshape(sv.shape[0], L, 2).sum(axis=2)

    if sv.ndim == 3 and sv.shape[1] == L and sv.shape[2] == 2:
        return sv.sum(axis=2)

    if sv.ndim == 3 and sv.shape[1] == L and sv.shape[2] == 1:
        return sv[..., 0]

    if sv.ndim == 4 and sv.shape[1] == L and sv.shape[2] == 2:
        return sv.sum(axis=(2, 3))

    raise ValueError(f"Unsupported SHAP shape for SNP matrix conversion: {sv.shape}, L={L}")

def _window_slice_from_model(windowed_ensemble, window_id: int) -> tuple[int, int]:
    """
    Compute (start,end) like the training WindowedEnsemble does.

    Supports:
      - legacy overlap_windows (50% overlap when True)
      - new window_overlap (fixed overlap count) when present
    """
    ws = int(getattr(windowed_ensemble, "window_size"))
    # new style: window_overlap param
    if hasattr(windowed_ensemble, "window_overlap"):
        ov = int(getattr(windowed_ensemble, "window_overlap"))
        stride = ws - ov
        start = int(window_id * stride)
    else:
        # legacy: 50% overlap when overlap_windows True
        if getattr(windowed_ensemble, "overlap_windows", False):
            start = int((window_id / 2) * ws)
        else:
            start = int(window_id * ws)
    end = start + ws
    return start, end

def get_window_metadata(windowed_ensemble, window_id: int) -> dict:
    """
    Return metadata for a model window.

    Important:
    - window_id is the current model window id used by WindowedEnsemble.
    - In shuffled group-order experiments, this may not equal the original LD group id.
    - We try to recover original_window_id/global_start/global_end/bp range from:
        1) windowed_ensemble.window_metadata
        2) windowed_ensemble.params["custom_windows"]
        3) fallback to _window_bounds()
    """
    if not hasattr(windowed_ensemble, "window_ids"):
        raise AttributeError("windowed_ensemble has no window_ids")

    if window_id not in windowed_ensemble.window_ids:
        raise ValueError(
            f"window_id={window_id} not in model.window_ids "
            f"(len={len(windowed_ensemble.window_ids)})"
        )

    local_model_index = int(windowed_ensemble.window_ids.index(window_id))

    meta = {
        "model_window_id": int(window_id),
        "local_model_index": local_model_index,
        "original_window_id": int(window_id),
        "start": -1,
        "end": -1,
        "global_start": -1,
        "global_end": -1,
        "bp_start": -1,
        "bp_end": -1,
        "size": -1,
    }

    # Preferred: metadata saved by WindowedEnsemble
    if hasattr(windowed_ensemble, "window_metadata") and windowed_ensemble.window_metadata is not None:
        if local_model_index < len(windowed_ensemble.window_metadata):
            w = windowed_ensemble.window_metadata[local_model_index]
            meta.update({
                "original_window_id": int(w.get("window_id", window_id)),
                "start": int(w.get("start", -1)),
                "end": int(w.get("end", -1)),
                "global_start": int(w.get("global_start", -1)),
                "global_end": int(w.get("global_end", -1)),
                "bp_start": int(w.get("bp_start", -1)),
                "bp_end": int(w.get("bp_end", -1)),
                "size": int(w.get("size", -1)),
            })
            return meta

    # Fallback: recover from params["custom_windows"]
    params = getattr(windowed_ensemble, "params", {})
    custom_windows = params.get("custom_windows", None) if isinstance(params, dict) else None

    if custom_windows is not None and local_model_index < len(custom_windows):
        w = custom_windows[local_model_index]
        meta.update({
            "original_window_id": int(w.get("window_id", window_id)),
            "start": int(w.get("start", -1)),
            "end": int(w.get("end", -1)),
            "global_start": int(w.get("global_start", -1)),
            "global_end": int(w.get("global_end", -1)),
            "bp_start": int(w.get("bp_start", -1)),
            "bp_end": int(w.get("bp_end", -1)),
            "size": int(w.get("size", -1)),
        })
        return meta

    # Final fallback: only know current model bounds
    s, e = windowed_ensemble._window_bounds(window_id)
    meta.update({
        "start": int(s),
        "end": int(e),
        "global_start": int(s),
        "global_end": int(e - 1),
        "size": int(e - s),
    })
    return meta

def get_global_feature_window_ids(windowed_ensemble) -> list[int]:
    """
    Return window ids in the exact feature order seen by the global model.

    Normal mode:
        global feature 0 -> window_ids[0]

    global_input_order mode:
        global feature 0 -> window_ids[global_input_order[0]]
    """
    if not hasattr(windowed_ensemble, "window_ids"):
        raise AttributeError("windowed_ensemble has no window_ids")

    base_ids = np.asarray(windowed_ensemble.window_ids, dtype=int)

    # Prefer attribute saved in WindowedEnsemble
    order = getattr(windowed_ensemble, "global_input_order", None)

    # Fallback: recover from saved params
    if order is None:
        params = getattr(windowed_ensemble, "params", {})
        if isinstance(params, dict):
            order = params.get("global_input_order", None)

    if order is None:
        return base_ids.astype(int).tolist()

    order = np.asarray(order, dtype=int)

    if len(order) != len(base_ids):
        raise ValueError(
            f"global_input_order length mismatch: "
            f"len(order)={len(order)}, len(window_ids)={len(base_ids)}"
        )

    if sorted(order.tolist()) != list(range(len(base_ids))):
        raise ValueError(
            "global_input_order is not a valid permutation of window indices."
        )

    mapped_ids = base_ids[order]

    print("[DEBUG] global_input_order detected.")
    print("[DEBUG] first 20 base window_ids:", base_ids[:20].tolist())
    print("[DEBUG] first 20 global_input_order:", order[:20].tolist())
    print("[DEBUG] first 20 global feature window_ids:", mapped_ids[:20].tolist())

    return mapped_ids.astype(int).tolist()

def load_local_snp_permutations(load_from: str, model_name: str):
    """
    Load saved local SNP permutations for shuffle_mode='local_snp_order'.

    Expected path:
      <load_from>/<model_name>/shuffle_artifacts/local_snp_permutations.pkl
    """
    path = os.path.join(
        load_from,
        model_name,
        "shuffle_artifacts",
        "local_snp_permutations.pkl"
    )

    if not os.path.exists(path):
        print(f"[DEBUG] local_snp_permutations.pkl not found: {path}")
        return None

    with open(path, "rb") as f:
        perms = pickle.load(f)

    print(f"[DEBUG] loaded local SNP permutations from: {path}")
    print(f"[DEBUG] local SNP permutations type: {type(perms)}")

    try:
        if isinstance(perms, dict):
            print(f"[DEBUG] local SNP permutations dict len: {len(perms)}")
            print(f"[DEBUG] local SNP permutations dict keys sample: {list(perms.keys())[:20]}")
        elif isinstance(perms, (list, tuple)):
            print(f"[DEBUG] local SNP permutations list len: {len(perms)}")
            if len(perms) > 0:
                print(f"[DEBUG] local SNP permutations first item type: {type(perms[0])}")
                try:
                    print(f"[DEBUG] local SNP permutations first item len: {len(perms[0])}")
                except Exception:
                    print(f"[DEBUG] local SNP permutations first item value: {perms[0]}")
        else:
            arr = np.asarray(perms)
            print(f"[DEBUG] local SNP permutations array shape: {arr.shape}, dtype={arr.dtype}")
    except Exception as exc:
        print(f"[WARN] could not inspect local SNP permutations: {exc}")

    return perms


def get_local_snp_order_and_map(windowed_ensemble, window_id: int, s: int, e: int):
    """
    Return local SNP permutation and SNP labels for one local window.

    Returns:
      local_pos_order:
        Local positions used to reorder x[:, s:e, :].
        x_win = x_win_natural[:, local_pos_order, :]

      snp_index_map:
        SNP index corresponding to each SHAP/local input column.
    """
    L = int(e) - int(s)
    if L <= 0:
        raise ValueError(f"Invalid local window bounds: s={s}, e={e}")

    meta = get_window_metadata(windowed_ensemble, window_id)
    local_model_index = int(meta.get("local_model_index", -1))
    original_window_id = int(meta.get("original_window_id", window_id))
    global_start = int(meta.get("global_start", -1))

    def _extract_from_container(container, source_name: str):
        if container is None:
            return None

        if isinstance(container, dict):
            keys_to_try = [
                window_id,
                str(window_id),
                original_window_id,
                str(original_window_id),
                local_model_index,
                str(local_model_index),
            ]

            for k in keys_to_try:
                if k in container:
                    print(f"[DEBUG] local SNP permutation found in {source_name} using key={k!r}")
                    return container[k]

            print(f"[DEBUG] {source_name} is dict but no matching key found.")
            print(f"[DEBUG] tried keys: {keys_to_try}")
            print(f"[DEBUG] available keys sample: {list(container.keys())[:20]}")
            return None

        if isinstance(container, (list, tuple)):
            if 0 <= local_model_index < len(container):
                print(f"[DEBUG] local SNP permutation found in {source_name}[local_model_index={local_model_index}]")
                return container[local_model_index]

            if 0 <= int(window_id) < len(container):
                print(f"[DEBUG] local SNP permutation found in {source_name}[window_id={window_id}]")
                return container[int(window_id)]

            return None

        arr = np.asarray(container)

        if arr.ndim == 1 and arr.shape[0] == L:
            print(f"[DEBUG] local SNP permutation found in {source_name} as 1D array")
            return arr

        if arr.ndim >= 2:
            if 0 <= local_model_index < arr.shape[0]:
                print(f"[DEBUG] local SNP permutation found in {source_name}[local_model_index={local_model_index}, :]")
                return arr[local_model_index]

            if 0 <= int(window_id) < arr.shape[0]:
                print(f"[DEBUG] local SNP permutation found in {source_name}[window_id={window_id}, :]")
                return arr[int(window_id)]

        return None

    raw_order = None

    if hasattr(windowed_ensemble, "local_snp_permutations"):
        raw_order = _extract_from_container(
            getattr(windowed_ensemble, "local_snp_permutations"),
            "windowed_ensemble.local_snp_permutations"
        )

    if raw_order is None:
        for attr in [
            "local_snp_order",
            "local_snp_orders",
            "local_feature_order",
            "local_feature_orders",
            "snp_order",
            "snp_orders",
            "feature_order",
            "feature_orders",
        ]:
            if hasattr(windowed_ensemble, attr):
                raw_order = _extract_from_container(
                    getattr(windowed_ensemble, attr),
                    f"windowed_ensemble.{attr}"
                )
                if raw_order is not None:
                    break

    if raw_order is None:
        params = getattr(windowed_ensemble, "params", {})
        if isinstance(params, dict):
            for key in [
                "local_snp_permutations",
                "local_snp_order",
                "local_snp_orders",
                "local_feature_order",
                "local_feature_orders",
                "snp_order",
                "snp_orders",
                "feature_order",
                "feature_orders",
            ]:
                if key in params:
                    raw_order = _extract_from_container(
                        params[key],
                        f"windowed_ensemble.params[{key!r}]"
                    )
                    if raw_order is not None:
                        break

    if raw_order is None:
        print("[WARN] no local SNP permutation found; using natural order.")
        local_pos_order = np.arange(L, dtype=int)
        base = global_start if global_start >= 0 else int(s)
        snp_index_map = np.arange(base, base + L, dtype=int)
        return local_pos_order.tolist(), snp_index_map.tolist()

    raw_order = np.asarray(raw_order, dtype=int).reshape(-1)

    print(f"[DEBUG] raw local SNP order length: {len(raw_order)}")
    print(f"[DEBUG] raw local SNP order first 20: {raw_order[:20].tolist()}")
    print(f"[DEBUG] raw local SNP order min/max: {raw_order.min()}/{raw_order.max()}")

    if len(raw_order) != L:
        raise ValueError(
            f"local SNP permutation length mismatch for window_id={window_id}: "
            f"len(raw_order)={len(raw_order)}, expected L={L}, "
            f"slice=({s},{e}), global_start={global_start}, "
            f"local_model_index={local_model_index}, original_window_id={original_window_id}"
        )

    if len(set(raw_order.tolist())) != L:
        raise ValueError(
            f"local SNP permutation contains duplicates for window_id={window_id}."
        )

    if raw_order.min() >= 0 and raw_order.max() < L:
        local_pos_order = raw_order
        base = global_start if global_start >= 0 else int(s)
        snp_index_map = base + local_pos_order

        print("[DEBUG] interpreted local SNP permutation as LOCAL POSITIONS.")
        print(f"[DEBUG] base SNP index used for labels: {base}")
        print(f"[DEBUG] local_pos_order first 20: {local_pos_order[:20].tolist()}")
        print(f"[DEBUG] snp_index_map first 20: {snp_index_map[:20].tolist()}")

        return local_pos_order.astype(int).tolist(), snp_index_map.astype(int).tolist()

    if raw_order.min() >= int(s) and raw_order.max() < int(e):
        local_pos_order = raw_order - int(s)
        snp_index_map = raw_order

        print("[DEBUG] interpreted local SNP permutation as X_MODEL ABSOLUTE INDICES.")
        print(f"[DEBUG] local_pos_order first 20: {local_pos_order[:20].tolist()}")
        print(f"[DEBUG] snp_index_map first 20: {snp_index_map[:20].tolist()}")

        return local_pos_order.astype(int).tolist(), snp_index_map.astype(int).tolist()

    if global_start >= 0 and raw_order.min() >= global_start and raw_order.max() < global_start + L:
        local_pos_order = raw_order - global_start
        snp_index_map = raw_order

        print("[DEBUG] interpreted local SNP permutation as GLOBAL SNP INDICES.")
        print(f"[DEBUG] local_pos_order first 20: {local_pos_order[:20].tolist()}")
        print(f"[DEBUG] snp_index_map first 20: {snp_index_map[:20].tolist()}")

        return local_pos_order.astype(int).tolist(), snp_index_map.astype(int).tolist()

    raise ValueError(
        f"Cannot interpret local SNP permutation for window_id={window_id}. "
        f"raw min/max={raw_order.min()}/{raw_order.max()}, "
        f"s/e={s}/{e}, global_start={global_start}, L={L}. "
        f"First 20 raw_order={raw_order[:20].tolist()}"
    )


def get_local_snp_index_map(windowed_ensemble, window_id: int, s: int, e: int) -> list[int]:
    """
    Backward-compatible wrapper.
    """
    _, snp_index_map = get_local_snp_order_and_map(windowed_ensemble, window_id, s, e)
    return snp_index_map


def explain_windowed_local(windowed_ensemble, x: np.ndarray, window_id: int, n_bg: int, n_test: int | None,
                          save_results_to: str, name: str, result_types: dict,
                          shap_chunk_size: int = 8):
    print(f"[DEBUG] local result_types keys for {name}, window {window_id}: {list(result_types.keys())}")

    if not hasattr(windowed_ensemble, "window_ids") or not hasattr(windowed_ensemble, "local_models"):
        raise AttributeError("Loaded model does not look like WindowedEnsemble (missing window_ids/local_models).")

    if window_id not in windowed_ensemble.window_ids:
        raise ValueError(f"window_id={window_id} not in model.window_ids (len={len(windowed_ensemble.window_ids)})")

    idx = windowed_ensemble.window_ids.index(window_id)
    local_model = windowed_ensemble.local_models[idx]

    # Use the model's true window bounds.
    # This supports fixed windows, aligned windows, and custom LD-group windows.
    s, e = windowed_ensemble._window_bounds(window_id)

    # Metadata: important for shuffled group-order experiments.
    meta = get_window_metadata(windowed_ensemble, window_id)

    print(
        "[DEBUG] local window metadata:",
        {
            "model_window_id": meta["model_window_id"],
            "local_model_index": meta["local_model_index"],
            "original_window_id": meta["original_window_id"],
            "start": meta["start"],
            "end": meta["end"],
            "global_start": meta["global_start"],
            "global_end": meta["global_end"],
            "bp_start": meta["bp_start"],
            "bp_end": meta["bp_end"],
            "size": meta["size"],
        }
    )

    # Sanity check: _window_bounds and metadata should agree when metadata is available.
    if meta["start"] != -1 and meta["end"] != -1:
        if int(meta["start"]) != int(s) or int(meta["end"]) != int(e):
            print(
                f"[WARN] metadata start/end != _window_bounds: "
                f"metadata=({meta['start']},{meta['end']}), bounds=({s},{e})"
            )

    # Natural slice first.
    x_win_natural = util.slice_dataset(x, s, e)

    # For shuffle_mode='local_snp_order', reorder columns exactly as in training.
    local_pos_order, snp_index_map = get_local_snp_order_and_map(
        windowed_ensemble,
        window_id,
        s,
        e
    )

    x_win = x_win_natural[:, local_pos_order, :]

    print("[DEBUG] x_win_natural shape:", x_win_natural.shape)
    print("[DEBUG] x_win reordered shape:", x_win.shape)
    print("[DEBUG] local_pos_order first 20:", local_pos_order[:20])
    print("[DEBUG] snp_index_map first 20:", snp_index_map[:20])
    print("[DEBUG] snp_index_map is natural s:e?:", snp_index_map == list(range(int(s), int(e))))

    if len(local_pos_order) != x_win_natural.shape[1]:
        raise ValueError(
            f"local_pos_order length mismatch: len(local_pos_order)={len(local_pos_order)}, "
            f"x_win_natural.shape[1]={x_win_natural.shape[1]}"
        )

    if len(snp_index_map) != x_win.shape[1]:
        raise ValueError(
            f"snp_index_map length mismatch: len(snp_index_map)={len(snp_index_map)}, "
            f"x_win.shape[1]={x_win.shape[1]}"
        )

    # Save mapping CSV.
    out_map = os.path.join(save_results_to, f"{name}_local_w{window_id}_snp_mapping.csv")
    df_map = pd.DataFrame({
        "shap_column": np.arange(len(snp_index_map), dtype=int),
        "local_pos_order": np.asarray(local_pos_order, dtype=int),
        "snp_index": np.asarray(snp_index_map, dtype=int),
        "natural_slice_snp_index": np.arange(int(s), int(e), dtype=int),
    })

    if int(meta.get("global_start", -1)) >= 0:
        gs = int(meta["global_start"])
        df_map["natural_global_snp_index"] = np.arange(gs, gs + len(snp_index_map), dtype=int)

    df_map["is_same_as_natural_slice"] = (
        df_map["snp_index"].to_numpy() == df_map["natural_slice_snp_index"].to_numpy()
    )

    if "natural_global_snp_index" in df_map.columns:
        df_map["is_same_as_natural_global"] = (
            df_map["snp_index"].to_numpy() == df_map["natural_global_snp_index"].to_numpy()
        )

    df_map.to_csv(out_map, index=False)
    print(f"[Saved] {out_map}")

    x_bg = x_win[:n_bg]
    x_test = x_win[n_bg:] if n_test is None else x_win[n_bg:n_bg + n_test]

    sv = compute_shap_in_chunks(
        local_model, x_bg, x_test,
        chunk_size=shap_chunk_size,
        desc=f"{name} local window {window_id}"
    )

    # SNP-level signed SHAP matrix.
    # This is the single source of truth for raw_shap, importance, heatmap, and bar.
    sv_snp = shap_to_snp_matrix(sv, x_win.shape[1])
    mean_abs = np.mean(np.abs(sv_snp), axis=0)

    print(f"[DEBUG] sv_snp shape: {sv_snp.shape}")
    print(f"[DEBUG] mean_abs shape: {mean_abs.shape}")
    print(f"[DEBUG] len(snp_index_map)={len(snp_index_map)}, len(mean_abs)={len(mean_abs)}")

    if len(mean_abs) != len(snp_index_map):
        raise ValueError(
            f"Length mismatch in local explain: len(mean_abs)={len(mean_abs)} "
            f"but len(snp_index_map)={len(snp_index_map)} "
            f"(window_id={window_id}, slice=({s},{e}), sv_shape={np.asarray(sv).shape})"
        )

    # Save compact metadata.
    out_meta = os.path.join(save_results_to, f"{name}_local_w{window_id}_metadata.csv")
    pd.DataFrame([meta]).to_csv(out_meta, index=False)
    print(f"[Saved] {out_meta}")

    # Local SNP importance CSV.
    df_importance = pd.DataFrame({
        "model_window_id": meta["model_window_id"],
        "local_model_index": meta["local_model_index"],
        "original_window_id": meta["original_window_id"],
        "window_start": meta["start"],
        "window_end": meta["end"],
        "global_start": meta["global_start"],
        "global_end": meta["global_end"],
        "bp_start": meta["bp_start"],
        "bp_end": meta["bp_end"],
        "shap_column": np.arange(len(snp_index_map), dtype=int),
        "local_pos_order": np.asarray(local_pos_order, dtype=int),
        "snp_index": np.asarray(snp_index_map, dtype=int),
        "importance": mean_abs.astype(float),
    })

    if "importance" in result_types or "raw_results" in result_types:
        out_csv = os.path.join(save_results_to, f"{name}_local_w{window_id}_snp_importance.csv")
        df_importance.to_csv(out_csv, index=False)
        print(f"[Saved] {out_csv}")

        # Optional backward-compatible old filename.
        out_csv_old = os.path.join(save_results_to, f"{name}_local_w{window_id}_snp_raw_results.csv")
        df_importance.to_csv(out_csv_old, index=False)
        print(f"[Saved] {out_csv_old}  [compatibility copy]")

    # Local SNP raw SHAP CSV: samples x SNPs, signed values.
    if "raw_results" in result_types:
        cols = [f"snp_{i}" for i in snp_index_map]
        out_sv = os.path.join(save_results_to, f"{name}_local_w{window_id}_snp_raw_shap.csv")

        df_raw = pd.DataFrame(sv_snp, columns=cols)
        df_raw.insert(0, "sample_index", np.arange(sv_snp.shape[0], dtype=int))
        df_raw.to_csv(out_sv, index=False)

        print(f"[Saved] {out_sv}")

    if "bar" in result_types:
        out_bar = os.path.join(save_results_to, f"{name}_local_w{window_id}_snp_bar.png")
        feat_names = [f"SNP_{i}" for i in snp_index_map]
        save_bar(mean_abs, feat_names, out_bar, top_n=30, ylabel="mean(|SHAP|)")
        print(f"[Saved] {out_bar}")

    if "heatmap" in result_types:
        S = sv_snp.T
        feat_names = [f"SNP_{i}" for i in snp_index_map]
        out_png = os.path.join(save_results_to, f"{name}_local_w{window_id}_snp_heatmap.png")
        plot_shap_heatmap_like_example(
            S,
            feat_names,
            out_png,
            fx=None,
            max_display=50,
            colorbar_label="SHAP value (impact on model output)"
        )
        print(f"[Saved] {out_png}")

def explain_windowed_global(windowed_ensemble, x: np.ndarray, n_bg: int, n_test: int | None,
                            save_results_to: str, name: str, result_types: dict,
                            shap_chunk_size: int = 8):
    """
    Global explanation for WindowedEnsemble.
    Uses all test samples by default when n_test is None.
    Computes SHAP in chunks to avoid OOM.
    """
    if not hasattr(windowed_ensemble, "get_global_inputs") or not hasattr(windowed_ensemble, "global_model"):
        x_bg = x[:n_bg]
        x_test = x[n_bg:] if n_test is None else x[n_bg:n_bg + n_test]

        sv = compute_shap_in_chunks(
            windowed_ensemble, x_bg, x_test,
            chunk_size=shap_chunk_size,
            desc=f"{name} global(fallback)"
        )
        mean_abs = mean_abs_feature_from_shap(sv)

        fx = None
        try:
            if hasattr(windowed_ensemble, "predict"):
                fx = np.asarray(windowed_ensemble.predict(x_test)).reshape(-1)
        except Exception:
            fx = None

        window_ids = get_global_feature_window_ids(windowed_ensemble)
        return _save_window_results(
            name, window_ids, mean_abs, save_results_to, result_types,
            shap_test=sv, fx=fx, windowed_ensemble=windowed_ensemble
        )

    x_bg = x[:n_bg]
    x_test = x[n_bg:] if n_test is None else x[n_bg:n_bg + n_test]

    xg_bg = windowed_ensemble.get_global_inputs(x_bg)
    xg_test = windowed_ensemble.get_global_inputs(x_test)

    sv = compute_shap_in_chunks(
        windowed_ensemble.global_model, xg_bg, xg_test,
        chunk_size=shap_chunk_size,
        desc=f"{name} global"
    )
    mean_abs = mean_abs_feature_from_shap(sv)

    fx = None
    try:
        if hasattr(windowed_ensemble.global_model, "predict"):
            fx = np.asarray(windowed_ensemble.global_model.predict(xg_test)).reshape(-1)
        elif hasattr(windowed_ensemble.global_model, "model"):
            fx = np.asarray(windowed_ensemble.global_model.model.predict(xg_test)).reshape(-1)
    except Exception:
        fx = None

    window_ids = get_global_feature_window_ids(windowed_ensemble)
    return _save_window_results(
        name, window_ids, mean_abs, save_results_to, result_types,
        shap_test=sv, fx=fx, windowed_ensemble=windowed_ensemble
    )

def _save_window_results(name: str, window_ids: list, mean_abs: np.ndarray, save_results_to: str, result_types: dict,
                        shap_test: np.ndarray | None = None, fx: np.ndarray | None = None,
                        windowed_ensemble=None):
    """Save global(window-level) explanation artifacts.

    Global level: each feature == one current model window.
    For shuffled group-order experiments, current model window ids may not match original LD group ids.
    Therefore this function saves window metadata when available.
    """
    os.makedirs(save_results_to, exist_ok=True)

    mean_abs = np.asarray(mean_abs, dtype=float).reshape(-1)
    window_ids = get_global_feature_window_ids(windowed_ensemble)

    rows = []
    for global_feature_index, (wid, imp) in enumerate(zip(window_ids, mean_abs)):
        row = {
            "global_feature_index": int(global_feature_index),
            "window_id": int(wid),
            "importance": float(imp),
        }

        if windowed_ensemble is not None:
            try:
                meta = get_window_metadata(windowed_ensemble, int(wid))
                row.update({
                    "local_model_index": meta["local_model_index"],
                    "original_window_id": meta["original_window_id"],
                    "window_start": meta["start"],
                    "window_end": meta["end"],
                    "global_start": meta["global_start"],
                    "global_end": meta["global_end"],
                    "bp_start": meta["bp_start"],
                    "bp_end": meta["bp_end"],
                    "size": meta["size"],
                })
            except Exception as e:
                print(f"[WARN] Could not get metadata for window_id={wid}: {e}")

        rows.append(row)

    ranking_df = pd.DataFrame(rows)
    ranking_df = ranking_df.sort_values("importance", ascending=False).reset_index(drop=True)

    # importance CSV
    if "importance" in result_types or "raw_results" in result_types:
        out_csv = os.path.join(save_results_to, f"{name}_global_window_importance.csv")
        ranking_df.to_csv(out_csv, index=False)
        print(f"[Saved] {out_csv}")

    # raw SHAP matrix CSV
    if "raw_results" in result_types and shap_test is not None:
        sv = _unwrap_shap_values(shap_test)
        A = np.asarray(sv, dtype=float)

        n_windows = len(window_ids)

        if A.ndim > 2:
            A = np.squeeze(A)

        if A.ndim == 3:
            win_axes = [ax for ax, d in enumerate(A.shape) if d == n_windows]
            if win_axes:
                A = np.moveaxis(A, win_axes[0], -1)
            if A.ndim == 3:
                if A.shape[0] == n_windows:
                    A = A[0, :, :]
                elif A.shape[1] == n_windows:
                    A = A[:, 0, :]
                else:
                    A = A[..., 0]

        if A.ndim != 2:
            raise ValueError(f"Unexpected global SHAP shape after squeeze: {A.shape}")

        if A.shape[1] == n_windows:
            S_instances = A
        elif A.shape[0] == n_windows:
            S_instances = A.T
        else:
            raise ValueError(f"Cannot align global SHAP to n_windows={n_windows}: got {A.shape}")

        # Use richer column names if metadata is available.
        cols = []
        for global_feature_index, wid in enumerate(window_ids):
            if windowed_ensemble is not None:
                try:
                    meta = get_window_metadata(windowed_ensemble, int(wid))
                    cols.append(
                        f"G{len(cols)}_Window_{wid}_orig_{meta['original_window_id']}_"
                        f"bp_{meta['bp_start']}_{meta['bp_end']}"
                    )
                except Exception:
                    cols.append(f"Window_{wid}")
            else:
                cols.append(f"Window_{wid}")

        df_raw = pd.DataFrame(S_instances, columns=cols)

        if fx is not None:
            fx_arr = np.asarray(fx, dtype=float).reshape(-1)
            if fx_arr.shape[0] == df_raw.shape[0]:
                df_raw.insert(0, "f(x)", fx_arr)

        out_raw = os.path.join(save_results_to, f"{name}_global_window_raw_shap.csv")
        df_raw.to_csv(out_raw, index=False)
        print(f"[Saved] {out_raw}")

    # global bar plot
    if "bar" in result_types:
        out_bar = os.path.join(save_results_to, f"{name}_global_window_bar.png")

        feat_names = []
        for global_feature_index, wid in enumerate(window_ids):
            if windowed_ensemble is not None:
                try:
                    meta = get_window_metadata(windowed_ensemble, int(wid))
                    feat_names.append(
                        f"G{global_feature_index}|W{wid}|orig{meta['original_window_id']}|"
                        f"{meta['bp_start']}-{meta['bp_end']}"
                    )
                except Exception:
                    feat_names.append(f"Window_{wid}")
            else:
                feat_names.append(f"Window_{wid}")

        save_bar(mean_abs, feat_names, out_bar, top_n=None, ylabel="mean(|SHAP|)")
        print(f"[Saved] {out_bar}")

    # global heatmap
    if "heatmap" in result_types and shap_test is not None:
        sv = _unwrap_shap_values(shap_test)
        A = np.asarray(sv, dtype=float)
        print("[DEBUG] raw sv shape:", sv.shape)

        if A.ndim > 2:
            A = np.squeeze(A)

        n_windows = len(window_ids)

        if A.ndim == 2:
            if A.shape[1] == n_windows:
                S_instances = A
            elif A.shape[0] == n_windows:
                S_instances = A.T
            else:
                raise ValueError(f"Cannot align global SHAP to n_windows={n_windows}: got {A.shape}")

        elif A.ndim == 3:
            win_axes = [ax for ax, d in enumerate(A.shape) if d == n_windows]
            if win_axes:
                A = np.moveaxis(A, win_axes[0], -1)
            if A.ndim == 3:
                if A.shape[1] == n_windows:
                    A = A[:, 0, :]
                else:
                    A = A[..., 0]
            if A.shape[1] != n_windows:
                raise ValueError(f"Cannot align 3D global SHAP to n_windows={n_windows}: got {A.shape}")
            S_instances = A

        else:
            raise ValueError(f"Unexpected global SHAP shape: {A.shape}")

        shap_matrix = S_instances.T

        feat_names = []
        for global_feature_index, wid in enumerate(window_ids):
            if windowed_ensemble is not None:
                try:
                    meta = get_window_metadata(windowed_ensemble, int(wid))
                    feat_names.append(
                        f"G{global_feature_index}|W{wid}|orig{meta['original_window_id']}|"
                        f"{meta['bp_start']}-{meta['bp_end']}"
                    )
                except Exception:
                    feat_names.append(f"Window_{wid}")
            else:
                feat_names.append(f"Window_{wid}")

        fx_plot = None
        if fx is not None:
            fx_arr = np.asarray(fx, dtype=float).reshape(-1)
            if fx_arr.shape[0] == shap_matrix.shape[1]:
                fx_plot = fx_arr

        out_png = os.path.join(save_results_to, f"{name}_global_window_heatmap.png")
        plot_shap_heatmap_like_example(
            shap_matrix,
            feat_names,
            out_png,
            fx=fx_plot,
            max_display=None,
            colorbar_label="SHAP value (impact on model output)"
        )
        print(f"[Saved] {out_png}")

    return ranking_df


def explain_single_model(model, x: np.ndarray, n_bg: int, n_test: int | None, save_results_to: str,
                         name: str, load_from: str, use_snp_map: bool, result_types: dict,
                         shap_chunk_size: int = 8):
    """
    Explain a single model directly on SNP features (one-hot). If n_test is None,
    all test samples are explained. SHAP is computed in chunks.
    """
    x_bg_full = x[:n_bg]
    x_test_full = x[n_bg:] if n_test is None else x[n_bg:n_bg + n_test]

    if use_snp_map:
        mp = load_snp_map(load_from, name)
        snp_index_map = mp["snp_index_map"]
        idx = np.array(snp_index_map, dtype=int)
        x_bg_in = x_bg_full[:, idx, :]
        x_test_in = x_test_full[:, idx, :]
        Lsel = len(snp_index_map)
    else:
        x_bg_in = x_bg_full
        x_test_in = x_test_full
        Lsel = x.shape[1]
        snp_index_map = list(range(Lsel))

    sv = compute_shap_in_chunks(
        model, x_bg_in, x_test_in,
        chunk_size=shap_chunk_size,
        desc=f"{name} single-model"
    )
    mean_abs = mean_abs_snp_from_shap(sv, L=Lsel)

    if "raw_results" in result_types or "importance" in result_types:
        out_csv = os.path.join(save_results_to, f"{name}_snp_raw_results.csv")
        pd.DataFrame({"snp_index": snp_index_map, "importance": mean_abs.astype(float)}).to_csv(out_csv, index=False)
        print(f"[Saved] {out_csv}")

    if "heatmap" in result_types:
        out_png = os.path.join(save_results_to, f"{name}_snp_heatmap.png")
        sv_hm = np.asarray(sv)
        if sv_hm.ndim > 2:
            sv_hm = sv_hm.reshape(-1, sv_hm.shape[-1])
        if sv_hm.shape[1] == Lsel * 2:
            sv_snp = sv_hm.reshape(sv_hm.shape[0], Lsel, 2).mean(axis=2)
        elif sv_hm.shape[1] == Lsel:
            sv_snp = sv_hm
        else:
            sv_snp = sv_hm
        S = sv_snp.T
        feat_names = [f"SNP_{i}" for i in snp_index_map]
        plot_shap_heatmap_like_example(S, feat_names, out_png, fx=None, max_display=50)
        print(f"[Saved] {out_png}")

    if "bar" in result_types:
        out_bar = os.path.join(save_results_to, f"{name}_snp_bar.png")
        save_bar(mean_abs, snp_index_map, out_bar, top_n=30, ylabel="mean(|SHAP|)")
        print(f"[Saved] {out_bar}")


def main(explain_config: dict, explain_seed: int):
    np.random.seed(int(explain_seed))

    # ---- load background from train ----
    x_path_train = explain_config.get("x_train_zarr", f"{explain_config['dataset']}/x_train.zarr")
    x_train = zarr.open(x_path_train, mode="r")

    if x_train.ndim != 3 or x_train.shape[2] != 2:
        raise ValueError(f"Expected genotype-encoded x_train with shape (N, L, 2). Got {x_train.shape}")

    # ---- load test from x_test (what you want) ----
    x_path_test = explain_config.get("x_test_zarr", f"{explain_config['dataset']}/x_test.zarr")
    x_test = zarr.open(x_path_test, mode="r")
    if x_test.ndim != 3 or x_test.shape[2] != 2:
        raise ValueError(f"Expected genotype-encoded x_test with shape (N, L, 2). Got {x_test.shape}")
    
    y_test_path = explain_config.get("y_test_path", f"{explain_config['dataset']}/y_test.pkl")
    y_test = np.asarray(pd.read_pickle(y_test_path)).reshape(-1)

    n_bg = int(explain_config.get("n_background", 50))

    background_indices = explain_config.get("background_indices", None)

    if background_indices is not None:
        background_indices = [int(i) for i in background_indices]
        if len(background_indices) != n_bg:
            raise ValueError(
                f"len(background_indices)={len(background_indices)} does not match n_background={n_bg}"
            )
        if max(background_indices) >= x_train.shape[0] or min(background_indices) < 0:
            raise ValueError(
                f"background_indices out of range for x_train with shape {x_train.shape}"
            )
        x_bg_fixed = x_train[background_indices]
    else:
        x_bg_fixed = x_train[:n_bg]

    if x_train.shape[0] < n_bg:
        raise ValueError(
            f"Not enough background samples in x_train: "
            f"{x_train.shape[0]} < n_background {n_bg}"
        )

    n_test = explain_config.get("n_test", None)
    if n_test is not None:
        n_test = int(n_test)

    shap_chunk_size = int(explain_config.get("shap_chunk_size", 8))

    # sort test samples by label (AFR=0 first, EUR=1 last)
    order = np.argsort(y_test)
    x_test_sorted = x_test[order]
    y_test_sorted = y_test[order]

    # keep the full test set in x; optional slicing happens inside explain functions
    x = np.concatenate([x_bg_fixed, x_test_sorted], axis=0)

    print(
        f"[DEBUG] bg samples: {x_bg_fixed.shape[0]} | "
        f"full test count: {x_test_sorted.shape[0]} | "
        f"x total: {x.shape[0]}"
    )

    save_results_to = explain_config["save_results_to"]
    os.makedirs(save_results_to, exist_ok=True)

    explain_type = explain_config.get("explain_type", "SHAP")
    if explain_type.upper() != "SHAP":
        raise NotImplementedError("This script currently supports SHAP only (explain_type=SHAP).")

    feature_type = explain_config.get("feature_type", "snp").lower()
    result_types = explain_config.get("result_types", {"raw_results": {}, "heatmap": {}, "bar": {}})

    print("[DEBUG] result_types keys:", list(result_types.keys()))
    print("\n[DEBUG] models in config:")
    for i, mm in enumerate(explain_config["models"]):
        print(f"  {i}: {mm.get('name')}")

    for m in explain_config["models"]:
        name = m["name"]
        load_from = m["load_from"]
        use_snp_map = bool(m.get("use_snp_map", False))

        print(f"\n========== START MODEL: {name} ==========")
        print(f"[DEBUG] load_from = {load_from}")

        # model-level slice (must match training)
        x_model = x
        dataset_slice = m.get("dataset_slice", None)
        print(f"[DEBUG] dataset_slice = {dataset_slice}")

        if dataset_slice is not None:
            if not (isinstance(dataset_slice, (list, tuple)) and len(dataset_slice) == 2):
                raise ValueError(f"{name}: dataset_slice must be [start, end] or null")
            s, e = int(dataset_slice[0]), int(dataset_slice[1])
            x_model = util.slice_dataset(x_model, s, e)
            print(f"[DEBUG] x_model sliced shape = {x_model.shape} | slice=({s}, {e})")
        else:
            print(f"[DEBUG] x_model full shape = {x_model.shape}")

        print(f"[DEBUG] about to load model: {name}")
        model = util.load_model(load_from, name)
        print(f"[DEBUG] loaded model: {name}")
        print(f"[DEBUG] model type: {type(model)}")
        local_snp_permutations = load_local_snp_permutations(load_from, name)
        if local_snp_permutations is not None:
            setattr(model, "local_snp_permutations", local_snp_permutations)
            print("[DEBUG] attached local_snp_permutations to loaded model.")
        else:
            print("[DEBUG] no local_snp_permutations attached to loaded model.")

        # WindowedEnsemble path
        is_windowed = hasattr(model, "window_ids") and hasattr(model, "local_models")
        print(f"[DEBUG] is_windowed = {is_windowed}, feature_type = {feature_type}")

        if is_windowed and feature_type == "window":
            print(f"[DEBUG] entering explain_windowed_global for {name}")
            ranking_df = explain_windowed_global(
                model, x_model, n_bg, n_test, save_results_to, name, result_types,
                shap_chunk_size=shap_chunk_size
            )
            print(f"[DEBUG] finished explain_windowed_global for {name}")

            # local windows to explain: manual + auto topK
            locals_manual = m.get("locals_to_explain", []) or []
            auto_top = bool(m.get("auto_explain_top_locals", False))
            top_k = int(m.get("top_local_k", 0)) if auto_top else 0

            chosen = list(locals_manual)
            if top_k > 0:
                top_ids = ranking_df["window_id"].tolist()[:top_k]
                chosen.extend(top_ids)

            # de-duplicate while preserving order
            seen = set()
            chosen_unique = []
            for wid in chosen:
                if wid in seen:
                    continue
                seen.add(wid)
                chosen_unique.append(int(wid))

            print(f"[DEBUG] chosen_unique for {name} = {chosen_unique}")

            if chosen_unique:
                print(f"[Auto/Manual locals] {name}: explaining locals window_id={chosen_unique}")
                for wid in chosen_unique:
                    print(f"[DEBUG] start local explain {name}, window_id={wid}")
                    explain_windowed_local(
                        model, x_model, wid, n_bg, n_test, save_results_to, name, result_types,
                        shap_chunk_size=shap_chunk_size
                    )
                    print(f"[DEBUG] finished local explain {name}, window_id={wid}")

            print(f"========== FINISH MODEL: {name} ==========")
            continue

        print(f"[DEBUG] entering explain_single_model for {name}")
        explain_single_model(
            model, x_model, n_bg, n_test, save_results_to, name, load_from,
            use_snp_map=use_snp_map, result_types=result_types,
            shap_chunk_size=shap_chunk_size
        )
        print(f"========== FINISH MODEL: {name} ==========")

if __name__ == "__main__":
    exp_config = util.get_exp_config("Explain models (supports WindowedEnsemble global/local + single models).")

    if "explain_single" not in exp_config:
        raise KeyError("Config must contain 'explain_single'.")

    for i, cfg in enumerate(exp_config["explain_single"]):
        print(f"\n# -- Running Explain Configuration {i+1}/{len(exp_config['explain_single'])} -- #\n")
        main(cfg, exp_config.get("explain_seed", 1024))
