import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd

from scipy.signal import resample
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from tqdm import tqdm

from github_release.config import (
    BASE_INPUT_COLS,
    OUTLIER_COLS,
    PCA_INPUT_COLS,
    SAMPLE_ID_COL,
    STATIC_FEATURES,
    STR_FEATURES,
    TARGET_COL,
    DATE_FEATURES,
    FEATURES,
    PreprocessingConfig,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def detect_outliers_iqr(df, cols):
    df_out = df.copy()
    for col in cols:
        if col not in df_out.columns:
            continue
        if not np.issubdtype(df_out[col].dtype, np.number):
            continue

        q1 = df_out[col].quantile(0.05)
        q3 = df_out[col].quantile(0.95)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df_out = df_out[(df_out[col] >= lower) & (df_out[col] <= upper)]

    return df_out.reset_index(drop=True)


def encode_categorical(df, cat_cols):
    df_out = df.copy()
    encoders = {}

    for col in cat_cols:
        encoder = LabelEncoder()
        df_out[col] = encoder.fit_transform(df_out[col].astype(str))
        encoders[col] = encoder

    return df_out, encoders


def repeat_and_trim(arr, target_len):
    out = arr.copy()
    while out.shape[0] < target_len:
        out = np.concatenate([out, out], axis=0)
    return out[:target_len]


def minmax_normalize_2d(X, eps=1e-8):
    x_min = np.min(X, axis=0, keepdims=True)
    x_max = np.max(X, axis=0, keepdims=True)
    return (X - x_min) / (x_max - x_min + eps)


def exponential_decay_accumulation(diff_map, alpha=0.95):
    ema = np.zeros_like(diff_map, dtype=np.float32)
    ema[0] = diff_map[0]
    for t in range(1, diff_map.shape[0]):
        ema[t] = alpha * ema[t - 1] + (1 - alpha) * diff_map[t]
    return ema


def build_cluster_patterns(X, labels, out_len=112, alpha=0.95):
    clusters = []

    for label in np.unique(labels):
        cluster_values = X[labels == label]
        if len(cluster_values) < 2:
            continue
        clusters.append(resample(cluster_values, out_len))

    if len(clusters) < 2:
        return None, None

    clusters = np.array(clusters, dtype=np.float32)
    representative = np.mean(clusters, axis=0).astype(np.float32)
    diff = np.mean(np.abs(clusters - representative), axis=0).astype(np.float32)
    variation = exponential_decay_accumulation(diff, alpha=alpha)
    variation = minmax_normalize_2d(variation).astype(np.float32)
    return representative, variation


def encode_variation_image(variation_map, width=168, height=112, dpi=112):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(variation_map.reshape(variation_map.shape[0], -1)).reshape(
        variation_map.shape
    )

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(scaled, linewidth=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.tight_layout(pad=0)
    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(height, width, 3)
    plt.close(fig)
    return image


def preprocess_dataframe(df):
    encoded_df, encoders = encode_categorical(df, STR_FEATURES)
    filtered_df = detect_outliers_iqr(encoded_df, OUTLIER_COLS)

    transformer = QuantileTransformer(
        output_distribution="normal",
        random_state=42,
        n_quantiles=min(1000, max(10, len(filtered_df))),
    )
    filtered_df[PCA_INPUT_COLS] = transformer.fit_transform(filtered_df[PCA_INPUT_COLS])

    metadata = {
        "sample_id_col": SAMPLE_ID_COL,
        "target_col": TARGET_COL,
        "str_features": STR_FEATURES,
        "features": FEATURES,
        "date_features": DATE_FEATURES,
        "static_features": STATIC_FEATURES,
        "pca_input_cols": PCA_INPUT_COLS,
        "base_input_cols": BASE_INPUT_COLS,
        "outlier_cols": OUTLIER_COLS,
        "label_encoders": {col: encoder.classes_.tolist() for col, encoder in encoders.items()},
    }
    return filtered_df, metadata


def build_dataset_from_dataframe(df, config):
    X0_all, X1_all, y_all = [], [], []

    for _, group in tqdm(
        df.groupby(SAMPLE_ID_COL),
        desc=f"PC{config.n_components}_CL{config.n_clusters}",
    ):
        try:
            if len(group) <= config.min_group_size:
                continue

            y_value = np.float32(group[TARGET_COL].iloc[0])
            base = repeat_and_trim(group[BASE_INPUT_COLS].values, config.fixed_seq_len)
            pca_input = repeat_and_trim(group[PCA_INPUT_COLS].values, config.fixed_seq_len)

            if np.unique(pca_input, axis=0).shape[0] <= config.n_components:
                continue

            pca = PCA(n_components=config.n_components)
            pca_features = pca.fit_transform(pca_input)
            full = np.concatenate([base, pca_features], axis=1)

            if np.unique(full, axis=0).shape[0] < config.n_clusters:
                continue

            kmeans = KMeans(
                n_clusters=config.n_clusters,
                n_init=config.kmeans_n_init,
                random_state=config.random_seed,
            )
            labels = kmeans.fit_predict(full)

            representative, variation_map = build_cluster_patterns(
                full,
                labels,
                out_len=config.out_len,
                alpha=config.ema_alpha,
            )
            if representative is None:
                continue

            variation_image = encode_variation_image(variation_map)
            X0_all.append(representative.astype(np.float32))
            X1_all.append(variation_image.astype(np.uint8))
            y_all.append(y_value)
        except Exception:
            continue

    X0 = np.asarray(X0_all, dtype=np.float32)
    X1 = np.asarray(X1_all, dtype=np.uint8)
    y = np.asarray(y_all, dtype=np.float32).reshape(-1, 1)
    return X0, X1, y


def save_dataset(output_dir, X0, X1, y, metadata, config):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X0.npy"), X0)
    np.save(os.path.join(output_dir, "X1.npy"), X1)
    np.save(os.path.join(output_dir, "y.npy"), y)

    payload = {
        "preprocessing_config": {
            "fixed_seq_len": config.fixed_seq_len,
            "out_len": config.out_len,
            "ema_alpha": config.ema_alpha,
            "n_components": config.n_components,
            "n_clusters": config.n_clusters,
            "min_group_size": config.min_group_size,
            "kmeans_n_init": config.kmeans_n_init,
            "random_seed": config.random_seed,
        },
        "metadata": metadata,
        "shapes": {
            "X0": list(X0.shape),
            "X1": list(X1.shape),
            "y": list(y.shape),
        },
    }

    with open(os.path.join(output_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def preprocess_dataset(csv_file, output_root, config=None):
    config = config or PreprocessingConfig()
    output_dir = os.path.join(
        output_root,
        f"PC{config.n_components}_CL{config.n_clusters}",
    )

    df = pd.read_csv(csv_file)
    processed_df, metadata = preprocess_dataframe(df)
    X0, X1, y = build_dataset_from_dataframe(processed_df, config)
    save_dataset(output_dir, X0, X1, y, metadata, config)
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Build X0/X1/y dataset with fixed PCA/K settings.")
    parser.add_argument("--csv-file", required=True, help="Path to the source CSV file.")
    parser.add_argument("--output-root", required=True, help="Directory where X0/X1/y will be saved.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = preprocess_dataset(args.csv_file, args.output_root)
    print(f"[DONE] dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
