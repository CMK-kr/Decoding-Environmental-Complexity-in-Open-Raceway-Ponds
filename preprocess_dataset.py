import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from tqdm import tqdm

from config import (
    BASE_INPUT_COLUMNS,
    CATEGORICAL_COLUMNS,
    DATA_CSV_PATH,
    DATASET_DIR,
    FIXED_SEQ_LEN,
    IMAGE_DPI,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    N_CLUSTERS,
    N_COMPONENTS,
    OUT_LEN,
    OUTLIER_COLUMNS,
    PCA_INPUT_COLUMNS,
    SAMPLE_ID_COLUMN,
    TAG,
    TARGET_COLUMN,
)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def detect_outliers_iqr(df, columns, lower_quantile=0.05, upper_quantile=0.95, iqr_scale=1.5):
    filtered = df.copy()
    for column in columns:
        if column not in filtered.columns or not np.issubdtype(filtered[column].dtype, np.number):
            continue
        q1 = filtered[column].quantile(lower_quantile)
        q3 = filtered[column].quantile(upper_quantile)
        iqr = q3 - q1
        lower = q1 - (iqr_scale * iqr)
        upper = q3 + (iqr_scale * iqr)
        filtered = filtered[(filtered[column] >= lower) & (filtered[column] <= upper)]
    return filtered


def encode_categorical(df, categorical_columns):
    encoded = df.copy()
    for column in categorical_columns:
        encoded[column] = LabelEncoder().fit_transform(encoded[column].astype(str))
    return encoded


def repeat_and_trim(array, target_len):
    output = np.asarray(array)
    while output.shape[0] < target_len:
        output = np.concatenate([output, output], axis=0)
    return output[:target_len]


def minmax_normalize_2d(array):
    return (array - array.min()) / (array.max() - array.min() + 1e-8)


def exponential_decay_accumulation(diff_map, alpha=0.95):
    ema = np.zeros_like(diff_map)
    ema[0] = diff_map[0]
    for index in range(1, diff_map.shape[0]):
        ema[index] = alpha * ema[index - 1] + (1.0 - alpha) * diff_map[index]
    return ema


def build_cluster_patterns(X, labels, out_len=OUT_LEN):
    cluster_samples = []
    for label in np.unique(labels):
        cluster_sample = X[labels == label]
        if len(cluster_sample) < 2:
            continue
        cluster_samples.append(resample(cluster_sample, out_len))

    if len(cluster_samples) < 2:
        return None, None

    stacked = np.asarray(cluster_samples)
    representative = np.mean(stacked, axis=0)
    difference_map = np.mean(np.abs(stacked - representative), axis=0)
    ema = exponential_decay_accumulation(difference_map)
    return representative, minmax_normalize_2d(ema)


def img_encoder(X, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, dpi=IMAGE_DPI):
    scaler = StandardScaler()
    normalized = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(normalized, linewidth=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.tight_layout(pad=0)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(height, width, 3)
    plt.close(fig)
    return image


def transform_common_dataframe(df):
    transformed = encode_categorical(df, CATEGORICAL_COLUMNS)
    transformed = detect_outliers_iqr(transformed, OUTLIER_COLUMNS)

    quantile_transformer = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, len(transformed)),
    )
    transformed.loc[:, PCA_INPUT_COLUMNS] = quantile_transformer.fit_transform(transformed[PCA_INPUT_COLUMNS])
    return transformed


def preprocess_dataset(csv_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    preprocessed_df = transform_common_dataframe(df)
    preprocessed_df.to_csv(output_dir / "preprocessed_df.csv", index=False, encoding="utf-8-sig")

    X0_all = []
    X1_all = []
    y_all = []
    sample_ids = []
    log_rows = []

    for sample_key, group in tqdm(preprocessed_df.groupby(SAMPLE_ID_COLUMN), desc="Preprocess {}".format(TAG)):
        try:
            if len(group) <= 10:
                log_rows.append({"sampleID": sample_key, "status": "skip_small"})
                continue

            y_value = group[TARGET_COLUMN].iloc[0]
            base_input = repeat_and_trim(group[BASE_INPUT_COLUMNS].values, FIXED_SEQ_LEN)
            pca_input = repeat_and_trim(group[PCA_INPUT_COLUMNS].values, FIXED_SEQ_LEN)

            if np.unique(pca_input, axis=0).shape[0] <= N_COMPONENTS:
                log_rows.append({"sampleID": sample_key, "status": "skip_pca"})
                continue

            pca = PCA(n_components=N_COMPONENTS)
            pca_features = pca.fit_transform(pca_input)
            full_input = np.concatenate([base_input, pca_features], axis=1)

            if np.unique(full_input, axis=0).shape[0] < N_CLUSTERS:
                log_rows.append({"sampleID": sample_key, "status": "skip_kmeans"})
                continue

            kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=5, random_state=None)
            labels = kmeans.fit_predict(full_input)

            X0, X1_map = build_cluster_patterns(full_input, labels, out_len=OUT_LEN)
            if X0 is None or X1_map is None:
                log_rows.append({"sampleID": sample_key, "status": "skip_pattern"})
                continue

            X1 = img_encoder(X1_map)
            X0_all.append(X0)
            X1_all.append(X1)
            y_all.append(y_value)
            sample_ids.append(sample_key)
            log_rows.append({"sampleID": sample_key, "status": "success", "n_rows": len(group)})
        except Exception as exc:
            log_rows.append({"sampleID": sample_key, "status": "error", "error": str(exc)})

    X0_array = np.asarray(X0_all, dtype=np.float32)
    X1_array = np.asarray(X1_all)
    y_array = np.asarray(y_all, dtype=np.float32).reshape(-1, 1)
    sample_ids_array = np.asarray(sample_ids)

    np.save(output_dir / "X0.npy", X0_array)
    np.save(output_dir / "X1.npy", X1_array)
    np.save(output_dir / "y.npy", y_array)
    np.save(output_dir / "sample_ids.npy", sample_ids_array)

    pd.DataFrame(log_rows).to_csv(output_dir / "group_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "tag": TAG,
                "csv_path": str(Path(csv_path)),
                "X0_shape": str(X0_array.shape),
                "X1_shape": str(X1_array.shape),
                "y_shape": str(y_array.shape),
                "n_samples": int(len(y_array)),
            }
        ]
    ).to_csv(output_dir / "dataset_summary.csv", index=False, encoding="utf-8-sig")
    save_json(
        {
            "tag": TAG,
            "n_components": N_COMPONENTS,
            "n_clusters": N_CLUSTERS,
            "csv_path": str(Path(csv_path)),
            "output_dir": str(output_dir),
        },
        output_dir / "preprocess_config.json",
    )
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Generate X0, X1, and y for the final PC3_CL6 pipeline.")
    parser.add_argument(
        "--csv-path",
        default=str(DATA_CSV_PATH),
        help="Path to the raw sample_df.csv file",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATASET_DIR),
        help="Directory for generated X0/X1/y artifacts",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    preprocess_dataset(args.csv_path, args.output_dir)
    print("Preprocessing finished: {}".format(args.output_dir))


if __name__ == "__main__":
    main()
