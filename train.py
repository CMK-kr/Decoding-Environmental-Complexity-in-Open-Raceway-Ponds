import argparse
import json
import os

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from github_release.config import BEST_MODEL_PARAMS, TrainingConfig
from github_release.model import build_multimodal_model, set_seed


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_qcut_labels(y, q=5):
    y_flat = np.asarray(y).reshape(-1)
    unique_count = len(np.unique(y_flat))
    max_bins = min(int(q), unique_count)

    for n_bins in range(max_bins, 1, -1):
        try:
            bins = pd.qcut(y_flat, q=n_bins, labels=False, duplicates="drop")
            bins = np.asarray(bins)
            if len(np.unique(bins)) >= 2:
                return bins
        except Exception:
            continue
    return None


def load_dataset(dataset_dir):
    X0 = np.load(os.path.join(dataset_dir, "X0.npy"))
    X1 = np.load(os.path.join(dataset_dir, "X1.npy"))
    y = np.load(os.path.join(dataset_dir, "y.npy"))

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return X0, X1, y


def split_dataset(X0, X1, y, config):
    indices = np.arange(len(y))
    y_bins = safe_qcut_labels(y, q=config.n_bins)

    stratify_labels = y_bins if y_bins is not None else None
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.random_seed,
        shuffle=True,
        stratify=stratify_labels,
    )

    train_val_y = y[train_val_idx]
    train_val_bins = safe_qcut_labels(train_val_y, q=config.n_bins)
    stratify_train_val = train_val_bins if train_val_bins is not None else None

    val_ratio = config.val_size / (1.0 - config.test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        random_state=config.random_seed,
        shuffle=True,
        stratify=stratify_train_val,
    )

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


def scale_splits(X0, X1, y, split_indices):
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    train_X0 = X0[train_idx].astype(np.float32)
    val_X0 = X0[val_idx].astype(np.float32)
    test_X0 = X0[test_idx].astype(np.float32)

    x1_scaler = MinMaxScaler()
    train_X1 = x1_scaler.fit_transform(X1[train_idx].reshape(len(train_idx), -1)).reshape(X1[train_idx].shape)
    val_X1 = x1_scaler.transform(X1[val_idx].reshape(len(val_idx), -1)).reshape(X1[val_idx].shape)
    test_X1 = x1_scaler.transform(X1[test_idx].reshape(len(test_idx), -1)).reshape(X1[test_idx].shape)

    y_scaler = StandardScaler()
    train_y = y_scaler.fit_transform(y[train_idx])
    val_y = y_scaler.transform(y[val_idx])
    test_y = y_scaler.transform(y[test_idx])

    return (
        (train_X0, train_X1, train_y),
        (val_X0, val_X1, val_y),
        (test_X0, test_X1, test_y),
        x1_scaler,
        y_scaler,
    )


def smape_metric_np(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(
        np.mean(200.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps))
    )


def evaluate_predictions(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape_metric_np(y_true, y_pred),
    }


def save_scalers(output_dir, x1_scaler, y_scaler):
    np.savez(
        os.path.join(output_dir, "x1_scaler.npz"),
        min_=x1_scaler.min_,
        scale_=x1_scaler.scale_,
        data_min_=x1_scaler.data_min_,
        data_max_=x1_scaler.data_max_,
        data_range_=x1_scaler.data_range_,
    )
    np.savez(
        os.path.join(output_dir, "y_scaler.npz"),
        mean_=y_scaler.mean_,
        scale_=y_scaler.scale_,
        var_=y_scaler.var_,
    )


def train_model(dataset_dir, output_dir, config=None, model_params=None):
    config = config or TrainingConfig()
    model_params = model_params or BEST_MODEL_PARAMS

    os.makedirs(output_dir, exist_ok=True)
    set_seed(config.random_seed)

    X0, X1, y = load_dataset(dataset_dir)
    split_indices = split_dataset(X0, X1, y, config)
    train_split, val_split, test_split, x1_scaler, y_scaler = scale_splits(X0, X1, y, split_indices)

    train_X0, train_X1, train_y = train_split
    val_X0, val_X1, val_y = val_split
    test_X0, test_X1, test_y = test_split

    model = build_multimodal_model(train_X0.shape[1:], train_X1.shape[1:], model_params)
    checkpoint_path = os.path.join(output_dir, "best_model.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_lr,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        [train_X0, train_X1],
        train_y,
        validation_data=([val_X0, val_X1], val_y),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    val_pred_scaled = model.predict([val_X0, val_X1], verbose=0)
    test_pred_scaled = model.predict([test_X0, test_X1], verbose=0)

    val_pred = y_scaler.inverse_transform(val_pred_scaled)
    test_pred = y_scaler.inverse_transform(test_pred_scaled)
    val_true = y_scaler.inverse_transform(val_y)
    test_true = y_scaler.inverse_transform(test_y)

    val_metrics = evaluate_predictions(val_true, val_pred)
    test_metrics = evaluate_predictions(test_true, test_pred)

    save_json(history.history, os.path.join(output_dir, "training_history.json"))
    save_json(val_metrics, os.path.join(output_dir, "val_metrics.json"))
    save_json(test_metrics, os.path.join(output_dir, "test_metrics.json"))
    save_json(model_params, os.path.join(output_dir, "model_params.json"))
    save_json(
        {
            "train_size": int(len(split_indices["train"])),
            "val_size": int(len(split_indices["val"])),
            "test_size": int(len(split_indices["test"])),
            "dataset_dir": dataset_dir,
        },
        os.path.join(output_dir, "split_summary.json"),
    )
    save_scalers(output_dir, x1_scaler, y_scaler)

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": checkpoint_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the fixed multimodal model on one train/val/test split.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing X0.npy, X1.npy and y.npy.")
    parser.add_argument("--output-dir", required=True, help="Directory where model artifacts will be saved.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = train_model(args.dataset_dir, args.output_dir)
    print("[DONE] training finished")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
