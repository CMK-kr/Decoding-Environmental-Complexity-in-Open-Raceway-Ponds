import argparse
import gc
import json
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import (
    BATCH_SIZE,
    BEST_PARAMS,
    DATASET_DIR,
    EPOCHS,
    N_BINS,
    SPLIT_SEED,
    TEST_SIZE,
    TRAINING_ROOT,
)
from model import build_model, normalize_best_params, set_seed, smape_metric_np


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    X0 = np.load(dataset_dir / "X0.npy")
    X1 = np.load(dataset_dir / "X1.npy")
    y = np.load(dataset_dir / "y.npy")
    sample_ids_path = dataset_dir / "sample_ids.npy"
    sample_ids = np.load(sample_ids_path, allow_pickle=True) if sample_ids_path.exists() else np.arange(len(y))

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    return X0, X1, y, sample_ids


def build_y_bins(y, n_bins):
    y_flat = y.reshape(-1)
    binned = pd.qcut(y_flat, q=n_bins, labels=False, duplicates="drop")
    if pd.isna(binned).any():
        raise ValueError("Target binning produced NaN values.")
    return np.asarray(binned, dtype=int)


def make_train_test_split(y, n_bins, test_size, seed):
    y_binned = build_y_bins(y, n_bins)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y_binned)), y_binned))
    return train_idx, test_idx, y_binned


def scale_split_data(X0, X1, y, train_idx, test_idx):
    trainX0 = X0[train_idx]
    testX0 = X0[test_idx]
    trainX1 = X1[train_idx]
    testX1 = X1[test_idx]
    train_y = y[train_idx]
    test_y = y[test_idx]

    trainX0_scaled = trainX0.astype(np.float32)
    testX0_scaled = testX0.astype(np.float32)

    scaler1 = MinMaxScaler()
    trainX1_scaled = scaler1.fit_transform(trainX1.reshape(trainX1.shape[0], -1)).reshape(trainX1.shape).astype(
        np.float32
    )
    testX1_scaled = scaler1.transform(testX1.reshape(testX1.shape[0], -1)).reshape(testX1.shape).astype(np.float32)

    y_scaler = StandardScaler()
    train_y_scaled = y_scaler.fit_transform(train_y).astype(np.float32)
    test_y_scaled = y_scaler.transform(test_y).astype(np.float32)

    return {
        "trainX0": trainX0,
        "testX0": testX0,
        "trainX1": trainX1,
        "testX1": testX1,
        "train_y": train_y,
        "test_y": test_y,
        "trainX0_scaled": trainX0_scaled,
        "testX0_scaled": testX0_scaled,
        "trainX1_scaled": trainX1_scaled,
        "testX1_scaled": testX1_scaled,
        "train_y_scaled": train_y_scaled,
        "test_y_scaled": test_y_scaled,
        "y_scaler": y_scaler,
    }


def build_model_inputs(data):
    return [data["trainX0_scaled"], data["trainX1_scaled"]], [data["testX0_scaled"], data["testX1_scaled"]]


def evaluate_split(model, inputs, y_scaled, y_scaler):
    evaluated = model.evaluate(inputs, y_scaled, verbose=0)
    loss = float(evaluated[0])
    mse = float(evaluated[1]) if len(evaluated) > 1 else float("nan")
    scaled_mae = float(evaluated[2]) if len(evaluated) > 2 else float("nan")

    y_pred_scaled = model.predict(inputs, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_scaled)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    smape = smape_metric_np(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    explained_variance = explained_variance_score(y_true, y_pred)

    metrics = {
        "loss": loss,
        "mse": mse,
        "scaled_mae": scaled_mae,
        "mae": float(mae),
        "r2": float(r2),
        "rmse": float(rmse),
        "mape": float(mape),
        "smape": float(smape),
        "explained_variance": float(explained_variance),
    }
    predictions_df = pd.DataFrame({"y_true": y_true.reshape(-1), "y_pred": y_pred.reshape(-1)})
    return metrics, predictions_df


def train_final_model(dataset_dir, output_root):
    X0, X1, y, sample_ids = load_dataset(dataset_dir)
    train_idx, test_idx, y_binned = make_train_test_split(y, N_BINS, TEST_SIZE, SPLIT_SEED)
    data = scale_split_data(X0, X1, y, train_idx, test_idx)
    train_inputs, test_inputs = build_model_inputs(data)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "sample_index": train_idx.astype(int),
            "sample_id": sample_ids[train_idx],
            "y_true": y[train_idx].reshape(-1),
            "y_binned": y_binned[train_idx],
            "split": "train",
        }
    ).to_csv(output_root / "train_assignment.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "sample_index": test_idx.astype(int),
            "sample_id": sample_ids[test_idx],
            "y_true": y[test_idx].reshape(-1),
            "y_binned": y_binned[test_idx],
            "split": "test",
        }
    ).to_csv(output_root / "test_assignment.csv", index=False, encoding="utf-8-sig")

    tf.keras.backend.clear_session()
    set_seed(SPLIT_SEED)
    model = build_model(
        input_shape0=data["trainX0"].shape[1:],
        input_shape1=data["trainX1"].shape[1:],
        best_params=normalize_best_params(BEST_PARAMS),
    )

    callbacks = [
        EarlyStopping(monitor="val_rmse", patience=30, mode="min", restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=0),
    ]

    start_time = time()
    history = model.fit(
        train_inputs,
        data["train_y_scaled"],
        validation_data=(test_inputs, data["test_y_scaled"]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )
    elapsed_seconds = time() - start_time

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(output_root / "keras_history.csv", index=False, encoding="utf-8-sig")

    train_metrics, train_predictions = evaluate_split(model, train_inputs, data["train_y_scaled"], data["y_scaler"])
    test_metrics, test_predictions = evaluate_split(model, test_inputs, data["test_y_scaled"], data["y_scaler"])

    train_metrics.update({"split": "train", "mode": "x0_x1", "epochs_ran": int(len(history_df)), "elapsed_seconds": elapsed_seconds})
    test_metrics.update({"split": "test", "mode": "x0_x1", "epochs_ran": int(len(history_df)), "elapsed_seconds": elapsed_seconds})

    pd.DataFrame([train_metrics]).to_csv(output_root / "train_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([test_metrics]).to_csv(output_root / "test_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat([pd.DataFrame([train_metrics]), pd.DataFrame([test_metrics])], axis=0).to_csv(
        output_root / "split_metrics.csv", index=False, encoding="utf-8-sig"
    )
    train_predictions.to_csv(output_root / "train_predictions.csv", index=False, encoding="utf-8-sig")
    test_predictions.to_csv(output_root / "test_predictions.csv", index=False, encoding="utf-8-sig")

    save_json(
        {
            "mode": "x0_x1",
            "dataset_dir": str(Path(dataset_dir)),
            "split_seed": SPLIT_SEED,
            "test_size": TEST_SIZE,
            "n_bins": N_BINS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "best_params": normalize_best_params(BEST_PARAMS),
            "epochs_ran": int(len(history_df)),
            "elapsed_seconds": float(elapsed_seconds),
        },
        output_root / "run_config.json",
    )

    del model
    gc.collect()
    tf.keras.backend.clear_session()

    summary_row = pd.DataFrame(
        [
            {
                "mode": "x0_x1",
                "train_r2": train_metrics["r2"],
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "test_r2": test_metrics["r2"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_mape": test_metrics["mape"],
                "epochs_ran": int(len(history_df)),
                "elapsed_seconds": float(elapsed_seconds),
            }
        ]
    )
    summary_row.to_csv(output_root / "training_summary.csv", index=False, encoding="utf-8-sig")
    return summary_row


def parse_args():
    parser = argparse.ArgumentParser(description="Train the final X0 + X1 multimodal model with a single holdout split.")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR), help="Directory containing X0.npy, X1.npy, y.npy")
    parser.add_argument(
        "--output-dir",
        default=str(TRAINING_ROOT / "x0_x1_single_holdout"),
        help="Directory for training outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print("Training mode: x0_x1")
    train_final_model(args.dataset_dir, output_root)
    print("Training finished: {}".format(output_root))


if __name__ == "__main__":
    main()
