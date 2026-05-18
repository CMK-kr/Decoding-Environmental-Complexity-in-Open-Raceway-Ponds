import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from time import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import (
    BEST_SINGLE_MEMBER_ID,
    DATASET_DIR,
    DEFAULT_REPRO_OUTPUT_DIR,
    DEFAULT_REPRO_TEST_ASSIGNMENT,
    DEFAULT_REPRO_TRAIN_FULL_ASSIGNMENT,
    DISTRIBUTION_WEIGHT_CLUSTERS,
    DISTRIBUTION_WEIGHT_PCA_COMPONENTS,
    DISTRIBUTION_WEIGHT_SEED,
    ENSEMBLE_MANIFEST_NAME,
    FINAL_ENSEMBLE_MEMBERS,
    MAX_TRAINABLE_PARAMS,
    MODE_NAME,
)
from model import build_model, normalize_best_params, set_seed, smape_metric_np

tf.get_logger().setLevel("ERROR")
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass


class IdentityScaler:
    def fit(self, values):
        return self

    def transform(self, values):
        return np.asarray(values, dtype=np.float32).reshape(-1, 1)

    def inverse_transform(self, values):
        return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def make_jsonable(obj):
    if isinstance(obj, dict):
        return {str(key): make_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(make_jsonable(data), file, ensure_ascii=False, indent=2)


def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def configure_tensorflow_runtime():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return {
        "gpu_count": len(gpus),
        "gpu_names": [gpu.name for gpu in gpus],
        "built_with_cuda": bool(tf.test.is_built_with_cuda()),
    }


def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    required = ("X0.npy", "X1.npy", "y.npy", "sample_ids.npy")
    missing = [name for name in required if not (dataset_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Dataset is missing {} in {}. Run preprocess.py first.".format(missing, dataset_dir))

    X0 = np.load(dataset_dir / "X0.npy")
    X1 = np.load(dataset_dir / "X1.npy")
    y = np.load(dataset_dir / "y.npy")
    sample_ids = np.load(dataset_dir / "sample_ids.npy", allow_pickle=True).astype(str)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if len(X0) != len(X1) or len(X0) != len(y) or len(X0) != len(sample_ids):
        raise ValueError("Dataset length mismatch.")
    return X0, X1, y.astype(np.float32), sample_ids


def load_assignment_frame(path, split_name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("{} assignment file was not found: {}".format(split_name, path))
    frame = pd.read_csv(path)
    missing = {"sample_index", "sampleID"} - set(frame.columns)
    if missing:
        raise ValueError("{} is missing required columns: {}".format(path, sorted(missing)))
    frame = frame.copy()
    frame["sample_index"] = frame["sample_index"].astype(int)
    frame["sampleID"] = frame["sampleID"].astype(str)
    if "split" not in frame.columns:
        frame["split"] = split_name
    return frame


def validate_assignments(train_df, test_df, sample_ids):
    for name, frame in (("train_full", train_df), ("test", test_df)):
        if frame.empty:
            raise ValueError("{} assignment is empty.".format(name))
        if frame["sample_index"].duplicated().any():
            raise ValueError("{} assignment has duplicated sample_index values.".format(name))
        if frame["sampleID"].duplicated().any():
            raise ValueError("{} assignment has duplicated sampleID values.".format(name))
        indices = frame["sample_index"].to_numpy(dtype=int)
        if indices.min() < 0 or indices.max() >= len(sample_ids):
            raise ValueError("{} assignment contains out-of-range sample_index.".format(name))
        expected = frame["sampleID"].astype(str).to_numpy()
        actual = sample_ids[indices].astype(str)
        if np.any(expected != actual):
            raise ValueError("{} assignment does not match sample_ids.npy.".format(name))
    overlap = set(train_df["sampleID"].astype(str)) & set(test_df["sampleID"].astype(str))
    if overlap:
        raise ValueError("train_full and test assignments overlap: {}".format(sorted(overlap)[:10]))


def save_preprocessors(preprocessors, path):
    payload = {
        "scale_x0": False,
        "scale_y": False,
        "y_scaling": "identity_no_target_scaling",
        "target_transform": "identity",
        "x1_scaler_min": preprocessors["x1_scaler"].min_.tolist(),
        "x1_scaler_scale": preprocessors["x1_scaler"].scale_.tolist(),
        "x1_scaler_feature_range": list(preprocessors["x1_scaler"].feature_range),
    }
    save_json(payload, path)


def load_preprocessors(path):
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    x1_scaler = MinMaxScaler(feature_range=tuple(payload.get("x1_scaler_feature_range", (0, 1))))
    x1_scaler.min_ = np.asarray(payload["x1_scaler_min"], dtype=np.float64)
    x1_scaler.scale_ = np.asarray(payload["x1_scaler_scale"], dtype=np.float64)
    x1_scaler.n_features_in_ = int(len(x1_scaler.min_))
    x1_scaler.data_min_ = -x1_scaler.min_ / np.maximum(x1_scaler.scale_, 1e-12)
    x1_scaler.data_max_ = (1.0 - x1_scaler.min_) / np.maximum(x1_scaler.scale_, 1e-12)
    x1_scaler.data_range_ = x1_scaler.data_max_ - x1_scaler.data_min_
    x1_scaler.n_samples_seen_ = 1
    return {"x1_scaler": x1_scaler, "y_scaler": IdentityScaler(), "target_transform": "identity"}


def fit_preprocessors(X1, y, fit_idx):
    fit_idx = np.asarray(fit_idx, dtype=int)
    x1_scaler = MinMaxScaler()
    x1_fit = X1[fit_idx]
    x1_scaler.fit(x1_fit.reshape(x1_fit.shape[0], -1))
    y_scaler = IdentityScaler()
    y_scaler.fit(y[fit_idx])
    return {"x1_scaler": x1_scaler, "y_scaler": y_scaler, "target_transform": "identity"}


def transform_arrays(X0, X1, y, indices, preprocessors):
    indices = np.asarray(indices, dtype=int)
    X0_part = X0[indices].astype(np.float32)
    X1_part = X1[indices].astype(np.float32)
    y_part = y[indices].astype(np.float32)
    original_x1_shape = X1_part.shape
    X1_part = preprocessors["x1_scaler"].transform(X1_part.reshape(original_x1_shape[0], -1)).reshape(original_x1_shape).astype(np.float32)
    y_scaled = preprocessors["y_scaler"].transform(y_part).astype(np.float32)
    return [X0_part, X1_part], y_scaled


def count_trainable_params(model):
    return int(np.sum([np.prod(weight.shape) for weight in model.trainable_weights]))


def evaluate_predictions(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0),
        "smape": float(smape_metric_np(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
    }


def extract_sample_group(sample_id):
    return str(sample_id).split("_", 1)[0]


def normalized_inverse_frequency(values, max_weight):
    values = np.asarray(values, dtype=object)
    counts = pd.Series(values).value_counts()
    weights = np.asarray([1.0 / float(counts[value]) for value in values], dtype=np.float32)
    weights = weights / max(1e-8, float(np.mean(weights)))
    return np.clip(weights, 1.0 / max(1.0, float(max_weight)), float(max_weight)).astype(np.float32)


def build_target_bins(y_values, n_bins=5):
    try:
        bins = pd.qcut(np.asarray(y_values, dtype=float).reshape(-1), q=int(n_bins), labels=False, duplicates="drop")
        bins = np.asarray(bins, dtype=int)
    except ValueError:
        bins = np.zeros(len(y_values), dtype=int)
    return np.asarray(["b{}".format(value) for value in bins], dtype=object)


def normalized_target_extreme_weights(y_values, max_weight):
    y_values = np.asarray(y_values, dtype=float).reshape(-1)
    median = float(np.nanmedian(y_values))
    q05 = float(np.nanquantile(y_values, 0.05))
    q95 = float(np.nanquantile(y_values, 0.95))
    scale = max(1e-8, 0.5 * (q95 - q05))
    tail_score = np.abs(y_values - median) / scale
    weights = 1.0 + np.clip(tail_score, 0.0, 1.0)
    weights = weights / max(1e-8, float(np.mean(weights)))
    return np.clip(weights, 1.0 / max(1.0, float(max_weight)), float(max_weight)).astype(np.float32)


def build_covariate_distribution_labels(x0_values, n_clusters, seed):
    x0_values = np.asarray(x0_values, dtype=np.float32)
    n_samples = len(x0_values)
    features = np.concatenate(
        [
            np.nanmean(x0_values, axis=1),
            np.nanstd(x0_values, axis=1),
            np.nanmin(x0_values, axis=1),
            np.nanmax(x0_values, axis=1),
        ],
        axis=1,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = StandardScaler().fit_transform(features)
    n_components = min(DISTRIBUTION_WEIGHT_PCA_COMPONENTS, features.shape[1], max(1, n_samples - 1))
    embedding = PCA(n_components=n_components, random_state=seed).fit_transform(features) if n_components >= 2 else features[:, :1]
    cluster_count = min(max(2, int(n_clusters)), n_samples)
    labels = KMeans(n_clusters=cluster_count, n_init=10, random_state=seed).fit_predict(embedding)
    return np.asarray(["cov_c{}".format(int(value)) for value in labels], dtype=object)


def make_sample_weights(y_values, sample_ids, x0_values, mode, strength, max_weight):
    mode = str(mode).lower()
    strength = float(strength)
    if mode == "none" or strength <= 0:
        return None, {"mode": "none"}

    parts = []
    summaries = []
    if mode == "target_covariate":
        labels = build_target_bins(y_values)
        parts.append(normalized_inverse_frequency(labels, max_weight=max_weight))
        parts.append(normalized_target_extreme_weights(y_values, max_weight=max_weight))
        summaries.append({"part": "target_bins_and_extremes"})

    if mode in {"covariate_cluster", "target_covariate"}:
        labels = build_covariate_distribution_labels(
            x0_values,
            n_clusters=DISTRIBUTION_WEIGHT_CLUSTERS,
            seed=DISTRIBUTION_WEIGHT_SEED,
        )
        parts.append(normalized_inverse_frequency(labels, max_weight=max_weight))
        summaries.append({"part": "covariate_cluster"})

    if not parts:
        raise ValueError("Unsupported final sample_weight_mode: {}".format(mode))

    combined = np.ones(len(sample_ids), dtype=np.float32)
    for part in parts:
        combined *= part.astype(np.float32)
    combined = combined / max(1e-8, float(np.mean(combined)))
    weights = 1.0 + strength * (combined - 1.0)
    weights = np.clip(weights, 1.0 / max(1.0, float(max_weight)), float(max_weight)).astype(np.float32)
    weights = weights / max(1e-8, float(np.mean(weights)))
    return weights, {
        "mode": mode,
        "strength": float(strength),
        "max_weight": float(max_weight),
        "min": float(np.min(weights)),
        "mean": float(np.mean(weights)),
        "max": float(np.max(weights)),
        "parts": summaries,
    }


def fit_prediction_correction(y_true, y_pred, correction_type, strength):
    correction_type = str(correction_type).lower()
    strength = float(strength)
    if correction_type == "none" or strength <= 0:
        return {"type": "none", "strength": 0.0, "slope": 1.0, "intercept": 0.0}

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) < 2 or float(np.std(y_pred)) <= 1e-12:
        slope = 1.0
        intercept = float(np.mean(y_true - y_pred))
        correction_type = "mean_shift"
    elif correction_type == "mean_shift":
        slope = 1.0
        intercept = float(np.mean(y_true - y_pred))
    elif correction_type == "linear":
        slope, intercept = np.polyfit(y_pred, y_true, deg=1)
        slope = float(np.clip(slope, 0.5, 1.5))
        intercept = float(intercept)
    else:
        raise ValueError("Unsupported correction type: {}".format(correction_type))

    return {"type": correction_type, "strength": float(np.clip(strength, 0.0, 1.0)), "slope": slope, "intercept": intercept}


def apply_prediction_correction_values(y_pred, correction):
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    correction = correction or {"type": "none", "strength": 0.0, "slope": 1.0, "intercept": 0.0}
    strength = float(correction.get("strength", 0.0))
    if correction.get("type", "none") == "none" or strength <= 0:
        return y_pred
    adjusted = float(correction.get("slope", 1.0)) * y_pred + float(correction.get("intercept", 0.0))
    return ((1.0 - strength) * y_pred + strength * adjusted).astype(np.float32)


def predict_with_model(model, X0, X1, y, sample_ids, indices, preprocessors):
    inputs, y_scaled = transform_arrays(X0, X1, y, indices, preprocessors)
    pred_scaled = model.predict(inputs, verbose=0)
    y_true = preprocessors["y_scaler"].inverse_transform(y_scaled).reshape(-1)
    y_pred = preprocessors["y_scaler"].inverse_transform(pred_scaled).reshape(-1)
    return y_true, y_pred, sample_ids[np.asarray(indices, dtype=int)].astype(str)


def save_prediction_frame(sample_ids, y_true, y_pred, path):
    save_csv(
        pd.DataFrame(
            {
                "sampleID": sample_ids,
                "y_true": np.asarray(y_true).reshape(-1),
                "y_pred": np.asarray(y_pred).reshape(-1),
                "abs_error": np.abs(np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1)),
            }
        ),
        path,
    )


def train_member(member, X0, X1, y, sample_ids, train_idx, run_root, verbose):
    member_id = member["member_id"]
    member_dir = run_root / member_id
    member_dir.mkdir(parents=True, exist_ok=True)
    params = normalize_best_params(member["params"])

    tf.keras.backend.clear_session()
    set_seed(member["seed"])

    preprocessors = fit_preprocessors(X1, y, train_idx)
    train_order_seed = int(member["seed"]) + 1009
    order_rng = np.random.RandomState(train_order_seed)
    train_idx = np.asarray(train_idx, dtype=int)[order_rng.permutation(len(train_idx))]
    save_csv(
        pd.DataFrame({"train_order": np.arange(len(train_idx)), "sample_index": train_idx, "sampleID": sample_ids[train_idx]}),
        member_dir / "train_data_order.csv",
    )

    train_inputs, train_y = transform_arrays(X0, X1, y, train_idx, preprocessors)
    model = build_model(train_inputs[0].shape[1:], train_inputs[1].shape[1:], params)
    n_params = count_trainable_params(model)
    if n_params > MAX_TRAINABLE_PARAMS:
        raise ValueError("{} exceeds parameter cap: {} > {}".format(member_id, n_params, MAX_TRAINABLE_PARAMS))

    sample_weights, sample_weight_summary = make_sample_weights(
        y[train_idx],
        sample_ids[train_idx],
        X0[train_idx],
        mode=member["sample_weight_mode"],
        strength=member["sample_weight_strength"],
        max_weight=member["sample_weight_max"],
    )

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(str(member_dir / "keras_epoch_log.csv"), append=False),
    ]
    fit_kwargs = {
        "epochs": int(member["epochs"]),
        "batch_size": 32,
        "verbose": int(verbose),
        "shuffle": True,
        "callbacks": callbacks,
    }
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights

    print("Training {}: seed={}, epochs={}, loss={}, weights={}".format(member_id, member["seed"], member["epochs"], params.get("loss_type"), member["sample_weight_mode"]))
    start = time()
    history = model.fit(train_inputs, train_y, **fit_kwargs)
    elapsed = time() - start

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    save_csv(history_df, member_dir / "refit_keras_history.csv")

    y_true_train, y_pred_train_raw, train_sample_ids = predict_with_model(model, X0, X1, y, sample_ids, train_idx, preprocessors)
    correction = fit_prediction_correction(
        y_true_train,
        y_pred_train_raw,
        correction_type=member["prediction_correction"],
        strength=member["correction_strength"],
    )
    y_pred_train = apply_prediction_correction_values(y_pred_train_raw, correction)
    train_metrics = evaluate_predictions(y_true_train, y_pred_train)
    train_metrics.update(
        {
            "split": "train_full",
            "member_id": member_id,
            "prediction_correction_type": correction["type"],
            "prediction_correction_strength": correction["strength"],
            "prediction_correction_slope": correction["slope"],
            "prediction_correction_intercept": correction["intercept"],
            "raw_r2": evaluate_predictions(y_true_train, y_pred_train_raw)["r2"],
            "epochs_ran": int(len(history_df)),
            "elapsed_seconds": float(elapsed),
            "trainable_params": int(n_params),
        }
    )

    save_prediction_frame(train_sample_ids, y_true_train, y_pred_train, member_dir / "train_predictions.csv")
    save_csv(pd.DataFrame([train_metrics]), member_dir / "train_metrics.csv")
    save_json(params, member_dir / "member_params.json")
    save_preprocessors(preprocessors, member_dir / "preprocessors.json")
    model.save_weights(str(member_dir / "final_single_model.weights.h5"))

    record = {
        "member_id": member_id,
        "seed": int(member["seed"]),
        "epochs": int(member["epochs"]),
        "params": params,
        "output_dir": str(member_dir),
        "weights_file": "final_single_model.weights.h5",
        "preprocessors_file": "preprocessors.json",
        "prediction_correction": correction,
        "sample_weight_mode": member["sample_weight_mode"],
        "sample_weight_strength": float(member["sample_weight_strength"]),
        "sample_weight_max": float(member["sample_weight_max"]),
        "sample_weight_summary": sample_weight_summary,
    }
    save_json({"member": record, "train_metrics": train_metrics}, member_dir / "run_config.json")
    return record


def predict_saved_member(record, X0, X1, y, sample_ids, indices):
    member_dir = Path(record["output_dir"])
    preprocessors = load_preprocessors(member_dir / record["preprocessors_file"])
    inputs, y_scaled = transform_arrays(X0, X1, y, indices, preprocessors)
    model = build_model(inputs[0].shape[1:], inputs[1].shape[1:], record["params"])
    model.load_weights(str(member_dir / record["weights_file"]))
    pred_scaled = model.predict(inputs, verbose=0)
    y_true = preprocessors["y_scaler"].inverse_transform(y_scaled).reshape(-1)
    y_pred_raw = preprocessors["y_scaler"].inverse_transform(pred_scaled).reshape(-1)
    y_pred = apply_prediction_correction_values(y_pred_raw, record["prediction_correction"])
    return y_true, y_pred, sample_ids[np.asarray(indices, dtype=int)].astype(str)


def evaluate_final_members(records, X0, X1, y, sample_ids, test_idx, run_root):
    all_predictions = []
    y_true = None
    test_sample_ids = None
    for record in records:
        yt, yp, ids = predict_saved_member(record, X0, X1, y, sample_ids, test_idx)
        if y_true is None:
            y_true = yt
            test_sample_ids = ids
        all_predictions.append(yp)

    ensemble_pred = np.mean(np.stack(all_predictions, axis=0), axis=0)
    ensemble_metrics = evaluate_predictions(y_true, ensemble_pred)
    ensemble_metrics.update({"inference_mode": "ensemble", "members_used": ",".join([record["member_id"] for record in records])})
    save_prediction_frame(test_sample_ids, y_true, ensemble_pred, run_root / "test_predictions_ensemble.csv")

    best_record = next(record for record in records if record["member_id"] == BEST_SINGLE_MEMBER_ID)
    single_index = records.index(best_record)
    single_pred = all_predictions[single_index]
    single_metrics = evaluate_predictions(y_true, single_pred)
    single_metrics.update({"inference_mode": "single", "member_id": BEST_SINGLE_MEMBER_ID, "members_used": BEST_SINGLE_MEMBER_ID})
    save_prediction_frame(test_sample_ids, y_true, single_pred, run_root / "test_predictions_single.csv")
    return ensemble_metrics, single_metrics


def main():
    parser = argparse.ArgumentParser(description="Train the fixed ensemble, fit train-based correction, and write test outputs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_REPRO_OUTPUT_DIR))
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    runtime = configure_tensorflow_runtime()
    X0, X1, y, sample_ids = load_dataset(DATASET_DIR)
    train_df = load_assignment_frame(DEFAULT_REPRO_TRAIN_FULL_ASSIGNMENT, "train_full")
    test_df = load_assignment_frame(DEFAULT_REPRO_TEST_ASSIGNMENT, "test")
    validate_assignments(train_df, test_df, sample_ids)
    train_idx = train_df["sample_index"].to_numpy(dtype=int)
    test_idx = test_df["sample_index"].to_numpy(dtype=int)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_dir) / ("ensemble_run_" + timestamp)
    run_root.mkdir(parents=True, exist_ok=True)

    records = []
    for member in FINAL_ENSEMBLE_MEMBERS:
        records.append(train_member(member, X0, X1, y, sample_ids, train_idx, run_root, verbose=args.verbose))

    ensemble_metrics, single_metrics = evaluate_final_members(records, X0, X1, y, sample_ids, test_idx, run_root)
    manifest = {
        "created_at": timestamp,
        "mode": MODE_NAME,
        "run_root": str(run_root),
        "dataset_dir": str(DATASET_DIR),
        "train_full_assignment": str(DEFAULT_REPRO_TRAIN_FULL_ASSIGNMENT),
        "test_assignment": str(DEFAULT_REPRO_TEST_ASSIGNMENT),
        "test_used_for_correction": False,
        "correction_fit_split": "train_full",
        "default_inference_mode": "ensemble",
        "best_single_member_id": BEST_SINGLE_MEMBER_ID,
        "members": records,
        "test_metrics_ensemble": ensemble_metrics,
        "test_metrics_single": single_metrics,
        "tensorflow_runtime": runtime,
    }
    save_json(manifest, run_root / ENSEMBLE_MANIFEST_NAME)
    save_json(manifest, Path(__file__).resolve().parent / ENSEMBLE_MANIFEST_NAME)

    print("Training finished: {}".format(run_root))
    print("Wrote ensemble manifest and test prediction files.")


if __name__ == "__main__":
    main()
