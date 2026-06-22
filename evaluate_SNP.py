import util
import zarr
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def compute_metrics(y_true, y_prob, metric_names):
    """
    y_true: (N,) binary labels
    y_prob: (N,) predicted probabilities
    """
    y_pred = (y_prob >= 0.5).astype(int)

    results = {}
    for m in metric_names:
        if m == "accuracy":
            results[m] = float(accuracy_score(y_true, y_pred))
        elif m == "precision":
            results[m] = float(precision_score(y_true, y_pred, zero_division=0))
        elif m == "recall":
            results[m] = float(recall_score(y_true, y_pred, zero_division=0))
        elif m == "f1":
            results[m] = float(f1_score(y_true, y_pred, zero_division=0))
        elif m == "roc_auc":
            # ROC AUC needs probabilities and both classes present
            if len(np.unique(y_true)) < 2:
                results[m] = np.nan
            else:
                results[m] = float(roc_auc_score(y_true, y_prob))
        else:
            raise ValueError(f"Unknown metric: {m}")

    return results


def main(eval_config, random_seed):
    
    # Load TEST dataset
    x_path_test  = eval_config.get("x_test_zarr",  f"{eval_config['dataset']}/x_test.zarr")
    x = zarr.open(x_path_test, mode="r")
    y = pd.read_pickle(f"{eval_config['dataset']}/y_test.pkl")
    y = np.asarray(y).reshape(-1)   # force (N,)

    print("[Loaded TEST]")
    print("  x dtype/shape:", x.dtype, x.shape)
    print("  y dtype/shape:", y.dtype, y.shape)

    all_results = []
   # Evaluate each model
    for model_cfg in eval_config["models"]:
        model_name = model_cfg["name"]
        load_from = model_cfg["load_from"]

        print(f"\n[Evaluating] {model_name}")

        # ---- Slice dataset (supports 2D & 3D) ----
        dataset_slice = model_cfg.get("dataset_slice", None)

        if dataset_slice is None:
            start, end = None, None
            x_eval = x
        else:
            if not (isinstance(dataset_slice, (list, tuple)) and len(dataset_slice) == 2):
                raise ValueError("dataset_slice must be [start, end] or null")
            start, end = int(dataset_slice[0]), int(dataset_slice[1])
            x_eval = util.slice_dataset(x, start, end)

        # ---- Load full model (including internal local/global models) ----
        model = util.load_model(load_from, model_name)

        
        # Prediction
        # WindowedEnsemble.predict should return probabilities
        y_prob = model.predict(x_eval)
        y_prob = np.asarray(y_prob).reshape(-1)

        # Metrics
        metrics = compute_metrics(
            y_true=y,
            y_prob=y_prob,
            metric_names=eval_config["metrics"]
        )

        row = {
            "model": model_name,
            "dataset": eval_config["dataset"],
            "slice_start": start,
            "slice_end": end,
            **metrics
        }
        all_results.append(row)

        print("  Metrics:", metrics)

    
    # Save results
    results_df = pd.DataFrame(all_results)
    util.create_directory(eval_config["save_evaluation_to"])

    out_path = f"{eval_config['save_evaluation_to']}/evaluation_results.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\n[Saved evaluation results]")
    print(out_path)


if __name__ == "__main__":
    exp_config = util.get_exp_config("Evaluate trained models.")

    if "evaluate" not in exp_config:
        print("No evaluation configuration found. Skipping.")
    else:
        for i, eval_cfg in enumerate(exp_config["evaluate"]):
            print(f"\n# -- Running Evaluation Configuration {i+1}/{len(exp_config['evaluate'])} -- #\n")
            main(eval_cfg, exp_config["random_seed"])
