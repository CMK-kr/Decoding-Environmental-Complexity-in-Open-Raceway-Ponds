import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import QuantileTransformer, StandardScaler

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from config import (
    BASE_INPUT_COLUMNS,
    CATEGORICAL_COLUMNS,
    COVARIATE_SPLIT_CANDIDATE_SEED_COUNT,
    COVARIATE_SPLIT_CATEGORICAL_COLUMNS,
    COVARIATE_SPLIT_CLUSTERS,
    COVARIATE_SPLIT_NUMERIC_COLUMNS,
    COVARIATE_SPLIT_PCA_COMPONENTS,
    DATASET_DIR,
    DEFAULT_REPRO_TEST_ASSIGNMENT,
    DEFAULT_REPRO_TRAIN_ASSIGNMENT,
    DEFAULT_REPRO_VAL_ASSIGNMENT,

    RAW_DATA_CSV_PATH,
    FIXED_SEQ_LEN,
    IMAGE_DPI,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    N_BINS,
    N_CLUSTERS,
    N_COMPONENTS,
    OUT_LEN,
    OUTLIER_COLUMNS,
    PCA_INPUT_COLUMNS,
    PREPROCESS_SEED,
    SAMPLE_ID_COLUMN,
    SPLIT_CANDIDATE_SEED_COUNT,
    SPLIT_SEED,
    TAG,
    TARGET_COLUMN,
    TEST_SIZE,
    UFS_PRIMARY_SPLIT_WEIGHT,
    VAL_SIZE,
    VAL_SPLIT_SEED,
    UFS_TARGET_SPLIT_SEED,
)

INTERNAL_SPLIT_COLUMN = "__split"
OUTLIER_GROUP_COLUMN = "__outlier_group"
ASSIGNMENT_SAMPLE_ID_ALIASES = (SAMPLE_ID_COLUMN, "sampleID", "sample_id", "sampleid")
SEPARATED_BASE_INPUT_COLUMNS = [column for column in BASE_INPUT_COLUMNS if column not in CATEGORICAL_COLUMNS]
DEFAULT_CATEGORICAL_MODE = "normalized_integer_in_x0"


def make_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(make_jsonable(data), file, ensure_ascii=False, indent=2)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def validate_required_columns(df):
    required = set(BASE_INPUT_COLUMNS) | set(PCA_INPUT_COLUMNS) | {SAMPLE_ID_COLUMN, TARGET_COLUMN}
    missing = sorted([column for column in required if column not in df.columns])
    if missing:
        raise ValueError("Raw CSV is missing required columns: {}".format(", ".join(missing)))


def read_raw_csv_robust(csv_path):
    """Read raw sample_df.csv with common encodings and preserve quoted CSV records."""
    csv_path = Path(csv_path)
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(csv_path)


def add_derived_columns(df):
    out = df.copy()
    sample_id = out[SAMPLE_ID_COLUMN].astype(str)
    if "UFS_ID" in CATEGORICAL_COLUMNS and "UFS_ID" not in out.columns:
        out["UFS_ID"] = sample_id.str.extract(r"(UFS-\d+)", expand=False).fillna("UFS-UNKNOWN")
    return out


def get_assignment_sample_id_column(df):
    normalized = {str(column).lower(): column for column in df.columns}
    for alias in ASSIGNMENT_SAMPLE_ID_ALIASES:
        if alias in df.columns:
            return alias
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None


def read_assignment_sample_ids(path):
    path = Path(path)
    if not path.exists():
        return None, "missing"
    df = pd.read_csv(path)
    sample_id_column = get_assignment_sample_id_column(df)
    if sample_id_column is None:
        return None, "missing_sample_id_column"
    sample_ids = tuple(df[sample_id_column].astype(str).tolist())
    return sample_ids, "loaded_sample_id"


def aggregate_target_for_sample(group):
    """Return one robust target per sampleID.

    Some sampleID groups contain multiple target rows.  Using iloc[0]/first()
    makes the target depend on row order and can create unstable split labels
    and unstable y values.  Median is deterministic and robust to duplicated or
    noisy target rows.
    """
    values = pd.to_numeric(group[TARGET_COLUMN], errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.median())


def target_summary_for_sample(group):
    values = pd.to_numeric(group[TARGET_COLUMN], errors="coerce").dropna()
    if values.empty:
        return {
            "target_unique_count": 0,
            "target_first": np.nan,
            "target_median": np.nan,
            "target_min": np.nan,
            "target_max": np.nan,
            "target_range": np.nan,
        }
    unique_values = np.unique(values.to_numpy(dtype=float))
    return {
        "target_unique_count": int(len(unique_values)),
        "target_first": float(values.iloc[0]),
        "target_median": float(values.median()),
        "target_min": float(values.min()),
        "target_max": float(values.max()),
        "target_range": float(values.max() - values.min()),
    }


def make_sample_summary(df):
    rows = []
    for sample_id, group in df.groupby(SAMPLE_ID_COLUMN, sort=True):
        target_value = aggregate_target_for_sample(group)
        if pd.isna(target_value):
            continue
        rows.append({SAMPLE_ID_COLUMN: str(sample_id), "target": float(target_value)})
    summary = pd.DataFrame(rows)
    if summary.empty or len(summary) < 2:
        raise ValueError("At least two valid sampleID groups with valid target values are required to create a train/test split.")
    summary[SAMPLE_ID_COLUMN] = summary[SAMPLE_ID_COLUMN].astype(str)
    summary["target"] = pd.to_numeric(summary["target"], errors="coerce")
    summary = summary.dropna(subset=["target"]).reset_index(drop=True)
    return summary


def build_robust_target_bins(y, requested_bins):
    y = np.asarray(y, dtype=float).reshape(-1)
    max_bins = min(int(requested_bins), len(np.unique(y)), max(2, len(y) // 2))
    for n_bins in range(max_bins, 1, -1):
        try:
            bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
            bins = np.asarray(bins, dtype=int)
            counts = pd.Series(bins).value_counts()
            if len(counts) >= 2 and int(counts.min()) >= 2:
                return bins, int(len(counts)), "stratified_qcut"
        except Exception:
            continue
    return None, 0, "random_fallback"


def make_seed_sample_split(df):
    sample_summary = make_sample_summary(df)
    sample_ids = sample_summary[SAMPLE_ID_COLUMN].astype(str).to_numpy()
    y = sample_summary["target"].to_numpy(dtype=float)
    bins, actual_bins, split_method = build_robust_target_bins(y, N_BINS)

    if bins is not None:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=SPLIT_SEED,
        )
        train_pos, test_pos = next(splitter.split(np.zeros(len(sample_ids)), bins))
    else:
        rng = np.random.RandomState(SPLIT_SEED)
        order = np.arange(len(sample_ids))
        rng.shuffle(order)
        n_test = max(1, int(round(len(order) * float(TEST_SIZE))))
        n_test = min(n_test, len(order) - 1)
        test_pos = np.sort(order[:n_test])
        train_pos = np.sort(order[n_test:])

    train_ids = tuple(sample_ids[train_pos].tolist())
    test_ids = tuple(sample_ids[test_pos].tolist())
    metadata = {
        "source": "seed_generated_sample_split",
        "split_method": split_method,
        "requested_bins": int(N_BINS),
        "actual_bins": int(actual_bins),
        "split_seed": int(SPLIT_SEED),
        "test_size": float(TEST_SIZE),
        "n_train_samples_requested": int(len(train_ids)),
        "n_test_samples_requested": int(len(test_ids)),
    }
    return train_ids, test_ids, metadata


def extract_sample_group(sample_id):
    text = str(sample_id)
    if "_" not in text:
        return text
    return text.split("_", 1)[0]


def build_local_target_bins(summary, n_bins):
    local_bins = np.zeros(len(summary), dtype=int)
    global_bins, _, _ = build_robust_target_bins(summary["target"].values, n_bins)
    if global_bins is None:
        global_bins = np.zeros(len(summary), dtype=int)

    for _, group in summary.groupby("sample_group", sort=True):
        indices = group.index.to_numpy()
        y_values = group["target"].to_numpy(dtype=float)
        max_bins = min(int(n_bins), len(np.unique(y_values)), max(1, len(group) // 2))
        bins = None
        for candidate_bins in range(max_bins, 1, -1):
            try:
                candidate = pd.qcut(y_values, q=candidate_bins, labels=False, duplicates="drop")
                candidate = np.asarray(candidate, dtype=int)
                counts = pd.Series(candidate).value_counts()
                if len(counts) >= 2 and int(counts.min()) >= 2:
                    bins = candidate
                    break
            except Exception:
                continue
        if bins is None:
            bins = np.zeros(len(group), dtype=int)
        local_bins[indices] = bins
    return local_bins, np.asarray(global_bins, dtype=int)


def collapse_rare_strata(labels, fallback_labels, min_count=2):
    labels = np.asarray(labels, dtype=object)
    fallback_labels = np.asarray(fallback_labels, dtype=object)
    collapsed = labels.copy()
    counts = pd.Series(labels).value_counts()
    for label, count in counts.items():
        if int(count) < int(min_count):
            collapsed[labels == label] = fallback_labels[labels == label]

    counts = pd.Series(collapsed).value_counts()
    rare_labels = set(counts[counts < int(min_count)].index)
    if rare_labels:
        collapsed = np.asarray(
            [fallback if label in rare_labels else label for label, fallback in zip(collapsed, fallback_labels)],
            dtype=object,
        )
    return collapsed


def distribution_l1(assignment, column, split_name):
    full = assignment[column].value_counts(normalize=True)
    split = assignment.loc[assignment["split"].eq(split_name), column].value_counts(normalize=True)
    labels = sorted(set(full.index).union(set(split.index)))
    return float(sum(abs(float(split.get(label, 0.0)) - float(full.get(label, 0.0))) for label in labels))


def split_moment_distance(assignment, split_name):
    y_all = assignment["target"].astype(float)
    y_split = assignment.loc[assignment["split"].eq(split_name), "target"].astype(float)
    global_std = float(y_all.std(ddof=0))
    if global_std <= 1e-12 or y_split.empty:
        return 0.0
    mean_distance = abs(float(y_split.mean()) - float(y_all.mean())) / global_std
    std_distance = abs(float(y_split.std(ddof=0)) - float(y_all.std(ddof=0))) / global_std
    return float(mean_distance + std_distance)


def score_split_assignment(assignment):
    split_names = tuple(sorted(assignment["split"].astype(str).unique()))
    target_l1 = np.mean([distribution_l1(assignment, "global_target_bin", split_name) for split_name in split_names])
    group_l1 = np.mean([distribution_l1(assignment, "sample_group", split_name) for split_name in split_names])
    strata_l1 = np.mean([distribution_l1(assignment, "stratify_key_raw", split_name) for split_name in split_names])
    moment_distance = np.mean([split_moment_distance(assignment, split_name) for split_name in split_names])
    return float(target_l1 + group_l1 + (0.5 * strata_l1) + (0.5 * moment_distance))


def stratified_or_random_split(indices, labels, test_size, seed):
    indices = np.asarray(indices)
    labels = np.asarray(labels, dtype=object)
    if len(indices) < 2:
        raise ValueError("At least two samples are required to split.")

    label_counts = pd.Series(labels).value_counts()
    can_stratify = len(label_counts) >= 2 and int(label_counts.min()) >= 2
    if can_stratify:
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(seed))
            left_pos, right_pos = next(splitter.split(np.zeros(len(indices)), labels))
            return indices[left_pos], indices[right_pos], "stratified"
        except ValueError:
            pass

    rng = np.random.RandomState(int(seed))
    order = np.arange(len(indices))
    rng.shuffle(order)
    n_right = max(1, int(round(len(order) * float(test_size))))
    n_right = min(n_right, len(order) - 1)
    right_pos = np.sort(order[:n_right])
    left_pos = np.sort(order[n_right:])
    return indices[left_pos], indices[right_pos], "random_fallback"


def first_non_null(series):
    series = series.dropna()
    if series.empty:
        return ""
    return str(series.iloc[0])


def build_covariate_sample_frame(df):
    """Build sample-level split features without using TARGET_COLUMN."""
    numeric_columns = [column for column in COVARIATE_SPLIT_NUMERIC_COLUMNS if column in df.columns and column != TARGET_COLUMN]
    categorical_columns = [column for column in COVARIATE_SPLIT_CATEGORICAL_COLUMNS if column in df.columns and column != TARGET_COLUMN]
    if not numeric_columns:
        raise ValueError("No numeric covariate columns are available for target-free splitting.")

    work = df[[SAMPLE_ID_COLUMN] + numeric_columns + categorical_columns].copy()
    work[SAMPLE_ID_COLUMN] = work[SAMPLE_ID_COLUMN].astype(str)
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    grouped_numeric = work.groupby(SAMPLE_ID_COLUMN, sort=True)[numeric_columns].agg(["mean", "std", "min", "max"])
    grouped_numeric.columns = ["{}_{}".format(column, stat) for column, stat in grouped_numeric.columns]
    grouped_numeric = grouped_numeric.reset_index()
    grouped_numeric["row_count"] = work.groupby(SAMPLE_ID_COLUMN, sort=True).size().values

    grouped_categorical = pd.DataFrame({SAMPLE_ID_COLUMN: grouped_numeric[SAMPLE_ID_COLUMN].astype(str)})
    for column in categorical_columns:
        values = work.groupby(SAMPLE_ID_COLUMN, sort=True)[column].apply(first_non_null).reset_index(drop=True)
        grouped_categorical[column] = values.astype(str)

    summary = grouped_numeric.merge(grouped_categorical, on=SAMPLE_ID_COLUMN, how="left")
    summary["sample_group"] = summary[SAMPLE_ID_COLUMN].map(extract_sample_group)
    return summary, numeric_columns, categorical_columns


def build_covariate_embedding(summary, numeric_feature_columns, categorical_columns):
    numeric = summary[numeric_feature_columns].replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median()).fillna(0.0)
    numeric = numeric.astype(np.float32)

    categorical_for_balance = list(categorical_columns) + ["sample_group"]
    categorical = pd.get_dummies(summary[categorical_for_balance].astype(str), dummy_na=True)
    features = pd.concat([numeric.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)
    features = features.fillna(0.0).astype(np.float32)

    scaled = StandardScaler().fit_transform(features)
    max_components = min(
        int(COVARIATE_SPLIT_PCA_COMPONENTS),
        scaled.shape[1],
        max(1, scaled.shape[0] - 1),
    )
    if max_components >= 2:
        pca = PCA(n_components=max_components, random_state=PREPROCESS_SEED)
        embedding = pca.fit_transform(scaled)
        explained = pca.explained_variance_ratio_.tolist()
    else:
        embedding = scaled[:, :1]
        explained = []

    cluster_count = min(int(COVARIATE_SPLIT_CLUSTERS), max(2, len(summary) // 20), len(summary))
    if cluster_count >= 2 and len(summary) >= cluster_count:
        kmeans = KMeans(n_clusters=cluster_count, n_init=10, random_state=PREPROCESS_SEED)
        clusters = kmeans.fit_predict(embedding)
        cluster_method = "kmeans_on_target_free_covariate_pca"
    else:
        clusters = np.zeros(len(summary), dtype=int)
        cluster_method = "single_cluster_fallback"

    embedded = summary.copy()
    embedded["covariate_cluster"] = np.asarray(clusters, dtype=int).astype(str)
    embedded["stratify_key_raw"] = embedded["sample_group"].astype(str) + "_c" + embedded["covariate_cluster"].astype(str)
    for index in range(embedding.shape[1]):
        embedded["covariate_pc{}".format(index + 1)] = embedding[:, index].astype(float)

    metadata = {
        "target_used_for_split": False,
        "numeric_columns": list(COVARIATE_SPLIT_NUMERIC_COLUMNS),
        "categorical_columns": list(COVARIATE_SPLIT_CATEGORICAL_COLUMNS),
        "feature_count_after_encoding": int(features.shape[1]),
        "pca_components": int(embedding.shape[1]),
        "pca_explained_variance_ratio": explained,
        "cluster_count": int(len(pd.Series(clusters).unique())),
        "cluster_method": cluster_method,
    }
    return embedded, metadata


def covariate_moment_distance(assignment, split_name, pc_columns):
    if not pc_columns:
        return 0.0
    split = assignment[assignment["split"].eq(split_name)]
    if split.empty:
        return 0.0
    distances = []
    for column in pc_columns:
        all_values = assignment[column].astype(float)
        split_values = split[column].astype(float)
        std = float(all_values.std(ddof=0))
        if std <= 1e-12:
            continue
        mean_distance = abs(float(split_values.mean()) - float(all_values.mean())) / std
        std_distance = abs(float(split_values.std(ddof=0)) - float(all_values.std(ddof=0))) / std
        distances.append(mean_distance + std_distance)
    if not distances:
        return 0.0
    return float(np.mean(distances))


def score_covariate_split_assignment(assignment):
    split_names = tuple(sorted(assignment["split"].astype(str).unique()))
    pc_columns = [column for column in assignment.columns if str(column).startswith("covariate_pc")]
    group_l1 = np.mean([distribution_l1(assignment, "sample_group", split_name) for split_name in split_names])
    cluster_l1 = np.mean([distribution_l1(assignment, "covariate_cluster", split_name) for split_name in split_names])
    strata_l1 = np.mean([distribution_l1(assignment, "stratify_key_raw", split_name) for split_name in split_names])
    moment_distance = np.mean([covariate_moment_distance(assignment, split_name, pc_columns) for split_name in split_names])
    secondary_score = float(cluster_l1 + (0.5 * strata_l1) + (0.5 * moment_distance))
    return float((float(UFS_PRIMARY_SPLIT_WEIGHT) * group_l1) + secondary_score)


def covariate_split_score_components(assignment):
    split_names = tuple(sorted(assignment["split"].astype(str).unique()))
    pc_columns = [column for column in assignment.columns if str(column).startswith("covariate_pc")]
    group_l1 = float(np.mean([distribution_l1(assignment, "sample_group", split_name) for split_name in split_names]))
    cluster_l1 = float(np.mean([distribution_l1(assignment, "covariate_cluster", split_name) for split_name in split_names]))
    strata_l1 = float(np.mean([distribution_l1(assignment, "stratify_key_raw", split_name) for split_name in split_names]))
    moment_distance = float(np.mean([covariate_moment_distance(assignment, split_name, pc_columns) for split_name in split_names]))
    secondary_score = float(cluster_l1 + (0.5 * strata_l1) + (0.5 * moment_distance))
    split_score = float((float(UFS_PRIMARY_SPLIT_WEIGHT) * group_l1) + secondary_score)
    return {
        "group_l1": group_l1,
        "cluster_l1": cluster_l1,
        "strata_l1": strata_l1,
        "moment_distance": moment_distance,
        "secondary_score": secondary_score,
        "split_score": split_score,
    }


def make_covariate_train_val_test_split(df):
    summary, numeric_columns, categorical_columns = build_covariate_sample_frame(df)
    numeric_feature_columns = [
        column
        for column in summary.columns
        if column not in {SAMPLE_ID_COLUMN, "sample_group"}
        and column not in set(categorical_columns)
    ]
    summary, embedding_metadata = build_covariate_embedding(summary, numeric_feature_columns, categorical_columns)
    fallback_labels = summary["sample_group"].astype(str)
    split_labels = fallback_labels.values
    indices = np.arange(len(summary))

    best_assignment = None
    best_score = None
    score_rows = []
    candidate_seeds = range(
        int(SPLIT_SEED),
        int(SPLIT_SEED) + max(1, int(COVARIATE_SPLIT_CANDIDATE_SEED_COUNT)),
    )
    val_fraction_of_train_val = float(VAL_SIZE) / max(1e-8, 1.0 - float(TEST_SIZE))

    for seed in candidate_seeds:
        train_val_pos, test_pos, test_method = stratified_or_random_split(
            indices,
            split_labels,
            test_size=float(TEST_SIZE),
            seed=int(seed),
        )
        train_val_labels = collapse_rare_strata(
            np.asarray(split_labels)[train_val_pos],
            fallback_labels.iloc[train_val_pos].values,
            min_count=2,
        )
        train_pos, val_pos, val_method = stratified_or_random_split(
            train_val_pos,
            train_val_labels,
            test_size=val_fraction_of_train_val,
            seed=int(seed) + int(VAL_SPLIT_SEED),
        )

        assignment = summary.copy()
        assignment["split"] = "train"
        assignment.loc[val_pos, "split"] = "val"
        assignment.loc[test_pos, "split"] = "test"
        score_components = covariate_split_score_components(assignment)
        score = score_components["split_score"]
        row = {
            "seed": int(seed),
            "val_seed": int(seed) + int(VAL_SPLIT_SEED),
            "split_score": float(score),
            "group_l1": float(score_components["group_l1"]),
            "cluster_l1": float(score_components["cluster_l1"]),
            "strata_l1": float(score_components["strata_l1"]),
            "pc_moment_distance": float(score_components["moment_distance"]),
            "secondary_score": float(score_components["secondary_score"]),
            "test_split_method": test_method,
            "val_split_method": val_method,
            "train_samples_requested": int(assignment["split"].eq("train").sum()),
            "val_samples_requested": int(assignment["split"].eq("val").sum()),
            "test_samples_requested": int(assignment["split"].eq("test").sum()),
        }
        score_rows.append(row)
        if best_score is None or score < best_score:
            best_score = float(score)
            best_assignment = assignment

    train_ids = tuple(best_assignment.loc[best_assignment["split"].eq("train"), SAMPLE_ID_COLUMN].tolist())
    val_ids = tuple(best_assignment.loc[best_assignment["split"].eq("val"), SAMPLE_ID_COLUMN].tolist())
    test_ids = tuple(best_assignment.loc[best_assignment["split"].eq("test"), SAMPLE_ID_COLUMN].tolist())
    best_row = min(score_rows, key=lambda row: row["split_score"])
    metadata = {
        "source": "target_free_covariate_train_val_test_distribution_split",
        "split_method": "ufs_primary_pc3_cluster_secondary_best_distribution_seed",
        "target_used_for_split": False,
        "primary_split_label": "sample_group",
        "secondary_split_balance": "PC1-PC3 covariate_cluster",
        "ufs_primary_split_weight": float(UFS_PRIMARY_SPLIT_WEIGHT),
        "split_seed": int(best_row["seed"]),
        "val_split_seed": int(best_row["val_seed"]),
        "candidate_seed_start": int(SPLIT_SEED),
        "candidate_seed_count": int(COVARIATE_SPLIT_CANDIDATE_SEED_COUNT),
        "train_size": float(1.0 - float(TEST_SIZE) - float(VAL_SIZE)),
        "val_size": float(VAL_SIZE),
        "test_size": float(TEST_SIZE),
        "selected_split_score": float(best_score),
        "selected_split_score_row": best_row,
        "split_candidate_scores": score_rows,
        "embedding_metadata": embedding_metadata,
        "n_train_samples_requested": int(len(train_ids)),
        "n_val_samples_requested": int(len(val_ids)),
        "n_test_samples_requested": int(len(test_ids)),
        "split_distribution": best_assignment["split"].value_counts().to_dict(),
    }
    return train_ids, val_ids, test_ids, metadata


def make_ufs_target_train_val_test_split(df):
    summary = make_sample_summary(df)
    summary["sample_group"] = summary[SAMPLE_ID_COLUMN].map(extract_sample_group)
    local_bins, global_bins = build_local_target_bins(summary, N_BINS)
    summary["local_target_bin"] = local_bins
    summary["global_target_bin"] = global_bins
    summary["stratify_key_raw"] = summary["sample_group"].astype(str) + "_b" + summary["local_target_bin"].astype(str)
    fallback_labels = summary["sample_group"].astype(str) + "_g" + summary["global_target_bin"].astype(str)
    split_labels = collapse_rare_strata(summary["stratify_key_raw"].values, fallback_labels.values, min_count=2)

    indices = np.arange(len(summary))
    train_val_pos, test_pos, test_method = stratified_or_random_split(
        indices,
        split_labels,
        test_size=float(TEST_SIZE),
        seed=int(SPLIT_SEED),
    )

    val_fraction_of_train_val = float(VAL_SIZE) / max(1e-8, 1.0 - float(TEST_SIZE))
    train_val_labels = collapse_rare_strata(
        np.asarray(split_labels)[train_val_pos],
        fallback_labels.iloc[train_val_pos].values,
        min_count=2,
    )
    train_pos, val_pos, val_method = stratified_or_random_split(
        train_val_pos,
        train_val_labels,
        test_size=val_fraction_of_train_val,
        seed=int(VAL_SPLIT_SEED),
    )

    assignment = summary.copy()
    assignment["split"] = "train"
    assignment.loc[val_pos, "split"] = "val"
    assignment.loc[test_pos, "split"] = "test"
    score = score_split_assignment(assignment)

    train_ids = tuple(assignment.loc[assignment["split"].eq("train"), SAMPLE_ID_COLUMN].tolist())
    val_ids = tuple(assignment.loc[assignment["split"].eq("val"), SAMPLE_ID_COLUMN].tolist())
    test_ids = tuple(assignment.loc[assignment["split"].eq("test"), SAMPLE_ID_COLUMN].tolist())
    metadata = {
        "source": "ufs_target_train_val_test_distribution_split",
        "split_method": "sample_group_local_target_bin_train_val_test",
        "requested_bins": int(N_BINS),
        "split_seed": int(SPLIT_SEED),
        "val_split_seed": int(VAL_SPLIT_SEED),
        "train_size": float(1.0 - float(TEST_SIZE) - float(VAL_SIZE)),
        "val_size": float(VAL_SIZE),
        "test_size": float(TEST_SIZE),
        "test_split_method": test_method,
        "val_split_method": val_method,
        "selected_split_score": float(score),
        "n_train_samples_requested": int(len(train_ids)),
        "n_val_samples_requested": int(len(val_ids)),
        "n_test_samples_requested": int(len(test_ids)),
        "split_distribution": assignment["split"].value_counts().to_dict(),
    }
    return train_ids, val_ids, test_ids, metadata


def make_ufs_target_sample_split(df):
    summary = make_sample_summary(df)
    summary["sample_group"] = summary[SAMPLE_ID_COLUMN].map(extract_sample_group)
    local_bins, global_bins = build_local_target_bins(summary, N_BINS)
    summary["local_target_bin"] = local_bins
    summary["global_target_bin"] = global_bins
    summary["stratify_key_raw"] = summary["sample_group"].astype(str) + "_b" + summary["local_target_bin"].astype(str)
    fallback_labels = summary["sample_group"].astype(str) + "_g" + summary["global_target_bin"].astype(str)
    split_labels = collapse_rare_strata(summary["stratify_key_raw"].values, fallback_labels.values, min_count=2)

    indices = np.arange(len(summary))
    best_assignment = None
    best_score = None
    score_rows = []
    if UFS_TARGET_SPLIT_SEED is None:
        candidate_seeds = range(int(SPLIT_SEED), int(SPLIT_SEED) + int(SPLIT_CANDIDATE_SEED_COUNT))
    else:
        candidate_seeds = [int(UFS_TARGET_SPLIT_SEED)]

    for seed in candidate_seeds:
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=int(seed))
            train_pos, test_pos = next(splitter.split(np.zeros(len(indices)), split_labels))
        except ValueError:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=int(seed))
            train_pos, test_pos = next(splitter.split(np.zeros(len(indices)), summary["global_target_bin"].values))

        assignment = summary.copy()
        assignment["split"] = "train"
        assignment.loc[test_pos, "split"] = "test"
        score = score_split_assignment(assignment)
        row = {
            "seed": int(seed),
            "split_score": float(score),
            "target_l1": float(np.mean([distribution_l1(assignment, "global_target_bin", split_name) for split_name in ("train", "test")])),
            "group_l1": float(np.mean([distribution_l1(assignment, "sample_group", split_name) for split_name in ("train", "test")])),
            "strata_l1": float(np.mean([distribution_l1(assignment, "stratify_key_raw", split_name) for split_name in ("train", "test")])),
            "moment_distance": float(np.mean([split_moment_distance(assignment, split_name) for split_name in ("train", "test")])),
        }
        score_rows.append(row)
        if best_score is None or score < best_score:
            best_score = float(score)
            best_assignment = assignment

    train_ids = tuple(best_assignment.loc[best_assignment["split"].eq("train"), SAMPLE_ID_COLUMN].tolist())
    test_ids = tuple(best_assignment.loc[best_assignment["split"].eq("test"), SAMPLE_ID_COLUMN].tolist())
    metadata = {
        "source": "ufs_target_distribution_split",
        "split_method": "sample_group_local_target_bin_best_distribution_seed",
        "requested_bins": int(N_BINS),
        "split_seed": int(best_assignment.attrs.get("seed", -1)),
        "candidate_seed_start": int(SPLIT_SEED),
        "candidate_seed_count": int(SPLIT_CANDIDATE_SEED_COUNT),
        "forced_split_seed": None if UFS_TARGET_SPLIT_SEED is None else int(UFS_TARGET_SPLIT_SEED),
        "selected_split_score": float(best_score),
        "n_train_samples_requested": int(len(train_ids)),
        "n_test_samples_requested": int(len(test_ids)),
        "split_candidate_scores": score_rows,
    }
    if score_rows:
        best_row = min(score_rows, key=lambda row: row["split_score"])
        metadata["split_seed"] = int(best_row["seed"])
        metadata["selected_split_score_row"] = best_row
    return train_ids, test_ids, metadata


def resolve_sample_split(df, split_source, train_assignment, test_assignment):
    split_source = str(split_source).lower()
    raw_sample_ids = set(df[SAMPLE_ID_COLUMN].astype(str).unique())

    if split_source in {"auto", "assignment"}:
        train_ids, train_status = read_assignment_sample_ids(train_assignment)
        test_ids, test_status = read_assignment_sample_ids(test_assignment)
        if train_ids is not None and test_ids is not None:
            train_set = set(train_ids)
            test_set = set(test_ids)
            overlap = train_set.intersection(test_set)
            if overlap:
                raise ValueError("Train/test assignment sampleIDs overlap: {}".format(sorted(overlap)[:10]))
            missing = sorted((train_set | test_set) - raw_sample_ids)
            if missing:
                raise ValueError(
                    "Assignment files contain sampleIDs that are not in the raw CSV: {}".format(missing[:10])
                )
            metadata = {
                "source": "assignment_sample_id",
                "train_assignment": str(train_assignment),
                "test_assignment": str(test_assignment),
                "n_train_samples_requested": int(len(train_ids)),
                "n_val_samples_requested": 0,
                "n_test_samples_requested": int(len(test_ids)),
            }
            return train_ids, tuple(), test_ids, metadata

        if split_source == "assignment":
            raise FileNotFoundError(
                "Usable assignment files were not found. train_assignment_status={}, "
                "test_assignment_status={}. Assignment CSVs must contain a sampleID column for leakage-safe preprocessing.".format(
                    train_status, test_status
                )
            )

    if split_source in {"covariate_tvt", "covariate_train_val_test", "target_free_covariate_tvt"}:
        return make_covariate_train_val_test_split(df)
    if split_source in {"ufs_target_tvt", "ufs_target_train_val_test", "ufs-target-tvt"}:
        return make_ufs_target_train_val_test_split(df)
    if split_source in {"ufs_target", "ufs-target", "ufs_target_distribution"}:
        train_ids, test_ids, metadata = make_ufs_target_sample_split(df)
        return train_ids, tuple(), test_ids, metadata
    train_ids, test_ids, metadata = make_seed_sample_split(df)
    return train_ids, tuple(), test_ids, metadata


def attach_split_column(df, train_sample_ids, val_sample_ids, test_sample_ids):
    train_set = set(map(str, train_sample_ids))
    val_set = set(map(str, val_sample_ids))
    test_set = set(map(str, test_sample_ids))
    if train_set.intersection(val_set) or train_set.intersection(test_set) or val_set.intersection(test_set):
        raise ValueError("Train/val/test sample sets overlap.")

    out = df.copy()
    sid = out[SAMPLE_ID_COLUMN].astype(str)
    out[INTERNAL_SPLIT_COLUMN] = np.where(
        sid.isin(train_set),
        "train",
        np.where(sid.isin(val_set), "val", np.where(sid.isin(test_set), "test", "unused")),
    )
    out = out[out[INTERNAL_SPLIT_COLUMN].isin(["train", "val", "test"])].copy()
    if out.empty:
        raise ValueError("No rows remain after applying train/val/test sample split.")
    return out


def fit_categorical_mappings(train_df, categorical_columns):
    mappings = {}
    for column in categorical_columns:
        if column not in train_df.columns:
            continue
        values = sorted(pd.Series(train_df[column].astype(str).unique()).dropna().tolist())
        mappings[column] = {value: index for index, value in enumerate(values)}
    return mappings


def apply_categorical_mappings(df, mappings, unknown_value=-1):
    encoded = df.copy()
    for column, mapping in mappings.items():
        encoded[column] = encoded[column].astype(str).map(mapping).fillna(unknown_value).astype(np.int32)
    return encoded


def build_categorical_feature_columns(categorical_mappings):
    columns = []
    for column in CATEGORICAL_COLUMNS:
        mapping = categorical_mappings.get(column, {})
        for value, index in sorted(mapping.items(), key=lambda item: item[1]):
            columns.append("{}={}".format(column, value))
    return columns


def make_categorical_onehot(row, categorical_mappings):
    values = []
    for column in CATEGORICAL_COLUMNS:
        mapping = categorical_mappings.get(column, {})
        encoded_value = int(row[column]) if column in row and pd.notna(row[column]) else -1
        onehot = np.zeros(len(mapping), dtype=np.float32)
        if 0 <= encoded_value < len(mapping):
            onehot[encoded_value] = 1.0
        values.extend(onehot.tolist())
    return np.asarray(values, dtype=np.float32)


def normalized_integer_categorical_frame(group, categorical_mappings):
    output = group[BASE_INPUT_COLUMNS].copy()
    for column in CATEGORICAL_COLUMNS:
        mapping = categorical_mappings.get(column, {})
        denominator = max(1, len(mapping) - 1)
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(-1).astype(np.float32)
        output[column] = output[column] / float(denominator)
    return output


def build_base_input_for_x0(group, x0_base_columns, categorical_mappings, categorical_mode):
    if categorical_mode == "integer_in_x0":
        return repeat_and_trim(group[BASE_INPUT_COLUMNS].values, FIXED_SEQ_LEN), None

    if categorical_mode == "normalized_integer_in_x0":
        normalized = normalized_integer_categorical_frame(group, categorical_mappings)
        return repeat_and_trim(normalized.values, FIXED_SEQ_LEN), None

    numeric_input = repeat_and_trim(group[x0_base_columns].values, FIXED_SEQ_LEN)
    x2 = make_categorical_onehot(group.iloc[0], categorical_mappings)

    if categorical_mode == "onehot_in_x0":
        categorical_input = np.repeat(x2.reshape(1, -1), FIXED_SEQ_LEN, axis=0)
        return np.concatenate([numeric_input, categorical_input], axis=1), x2

    if categorical_mode == "separate_x2":
        return numeric_input, x2

    raise ValueError("Unsupported categorical_mode: {}".format(categorical_mode))


def fit_iqr_bounds(train_df, columns, lower_quantile=0.05, upper_quantile=0.95, iqr_scale=1.5):
    bounds = {}
    for column in columns:
        if column not in train_df.columns or not np.issubdtype(train_df[column].dtype, np.number):
            continue
        series = pd.to_numeric(train_df[column], errors="coerce").dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(lower_quantile))
        q3 = float(series.quantile(upper_quantile))
        iqr = q3 - q1
        bounds[column] = {
            "lower_quantile": float(lower_quantile),
            "upper_quantile": float(upper_quantile),
            "iqr_scale": float(iqr_scale),
            "q1": q1,
            "q3": q3,
            "lower": float(q1 - (iqr_scale * iqr)),
            "upper": float(q3 + (iqr_scale * iqr)),
        }
    return bounds


def apply_iqr_bounds(df, bounds, apply_target_to_train_only=False):
    if not bounds:
        return df.copy(), pd.Series(True, index=df.index)

    keep_mask = pd.Series(True, index=df.index)
    for column, meta in bounds.items():
        if column not in df.columns:
            continue
        column_mask = df[column].between(meta["lower"], meta["upper"], inclusive="both")
        column_mask = column_mask.fillna(False)
        if apply_target_to_train_only and column == TARGET_COLUMN:
            train_rows = df[INTERNAL_SPLIT_COLUMN].eq("train")
            keep_mask &= (~train_rows) | column_mask
        else:
            keep_mask &= column_mask
    return df.loc[keep_mask].copy(), keep_mask


def fit_iqr_bounds_by_group(
    train_df,
    columns,
    group_column,
    lower_quantile=0.05,
    upper_quantile=0.95,
    iqr_scale=1.5,
    min_group_rows=50,
):
    global_bounds = fit_iqr_bounds(
        train_df,
        columns,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        iqr_scale=iqr_scale,
    )
    grouped_bounds = {}
    group_stats = []

    if group_column not in train_df.columns:
        return global_bounds, grouped_bounds, group_stats

    for group_name, group in train_df.groupby(group_column):
        group_name = str(group_name)
        n_rows = int(len(group))
        if n_rows < int(min_group_rows):
            grouped_bounds[group_name] = global_bounds
            group_stats.append({
                "group": group_name,
                "fit_rows": n_rows,
                "used_fallback": True,
                "reason": "below_min_group_rows",
            })
            continue

        bounds = fit_iqr_bounds(
            group,
            columns,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            iqr_scale=iqr_scale,
        )
        for column, meta in global_bounds.items():
            if column not in bounds:
                fallback_meta = dict(meta)
                fallback_meta["fallback_to_global"] = True
                bounds[column] = fallback_meta
        grouped_bounds[group_name] = bounds
        group_stats.append({
            "group": group_name,
            "fit_rows": n_rows,
            "used_fallback": False,
            "reason": "ufs_group_bounds",
        })

    return global_bounds, grouped_bounds, group_stats


def apply_iqr_bounds_by_group(
    df,
    global_bounds,
    grouped_bounds,
    group_column,
    apply_target_to_train_only=False,
):
    if not global_bounds and not grouped_bounds:
        return df.copy(), pd.Series(True, index=df.index)

    if group_column not in df.columns:
        return apply_iqr_bounds(
            df,
            global_bounds,
            apply_target_to_train_only=apply_target_to_train_only,
        )

    keep_mask = pd.Series(True, index=df.index)
    for group_name, group in df.groupby(group_column):
        group_name = str(group_name)
        bounds = grouped_bounds.get(group_name, global_bounds)
        if not bounds:
            continue

        group_keep = pd.Series(True, index=group.index)
        for column, meta in bounds.items():
            if column not in group.columns:
                continue
            column_mask = group[column].between(meta["lower"], meta["upper"], inclusive="both")
            column_mask = column_mask.fillna(False)
            if apply_target_to_train_only and column == TARGET_COLUMN:
                train_rows = group[INTERNAL_SPLIT_COLUMN].eq("train")
                group_keep &= (~train_rows) | column_mask
            else:
                group_keep &= column_mask
        keep_mask.loc[group.index] = group_keep

    return df.loc[keep_mask].copy(), keep_mask


def summarize_outlier_filter_by_group(df, keep_mask, group_column):
    if group_column not in df.columns:
        return []

    rows = []
    for group_name, group in df.groupby(group_column):
        kept = keep_mask.loc[group.index]
        train_rows = group[INTERNAL_SPLIT_COLUMN].eq("train")
        val_rows = group[INTERNAL_SPLIT_COLUMN].eq("val")
        test_rows = group[INTERNAL_SPLIT_COLUMN].eq("test")
        rows.append({
            "group": str(group_name),
            "rows_before": int(len(group)),
            "rows_after": int(kept.sum()),
            "rows_removed": int(len(group) - kept.sum()),
            "train_rows_before": int(train_rows.sum()),
            "train_rows_after": int((kept & train_rows).sum()),
            "val_rows_before": int(val_rows.sum()),
            "val_rows_after": int((kept & val_rows).sum()),
            "test_rows_before": int(test_rows.sum()),
            "test_rows_after": int((kept & test_rows).sum()),
        })
    return rows


def fit_quantile_transformer(train_df):
    if len(train_df) < 2:
        raise ValueError("At least two train rows are required to fit QuantileTransformer.")
    transformer = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, len(train_df)),
        random_state=PREPROCESS_SEED,
    )
    transformer.fit(train_df[PCA_INPUT_COLUMNS].astype(np.float32))
    return transformer


def transform_common_dataframe_leakage_safe(
    df,
    train_sample_ids,
    val_sample_ids,
    test_sample_ids,
    include_target_outlier=False,
    iqr_scale=1.5,
    outlier_scope="ufs",
    min_ufs_outlier_rows=50,
):
    split_df = attach_split_column(df, train_sample_ids, val_sample_ids, test_sample_ids)

    train_rows_raw = split_df[split_df[INTERNAL_SPLIT_COLUMN].eq("train")]
    categorical_mappings = fit_categorical_mappings(train_rows_raw, CATEGORICAL_COLUMNS)
    encoded = apply_categorical_mappings(split_df, categorical_mappings)
    outlier_scope = str(outlier_scope).strip().lower()
    if outlier_scope in {"ufs", "ufs_group", "ufs_id"}:
        encoded = encoded.copy()
        encoded[OUTLIER_GROUP_COLUMN] = encoded[SAMPLE_ID_COLUMN].map(extract_sample_group)

    outlier_columns = [column for column in OUTLIER_COLUMNS if column != TARGET_COLUMN]
    if include_target_outlier:
        outlier_columns = list(outlier_columns) + [TARGET_COLUMN]

    train_rows_encoded = encoded[encoded[INTERNAL_SPLIT_COLUMN].eq("train")]
    group_iqr_bounds = {}
    outlier_group_fit_stats = []
    if outlier_scope in {"ufs", "ufs_group", "ufs_id"}:
        iqr_bounds, group_iqr_bounds, outlier_group_fit_stats = fit_iqr_bounds_by_group(
            train_rows_encoded,
            outlier_columns,
            OUTLIER_GROUP_COLUMN,
            iqr_scale=iqr_scale,
            min_group_rows=int(min_ufs_outlier_rows),
        )
        filtered, keep_mask = apply_iqr_bounds_by_group(
            encoded,
            iqr_bounds,
            group_iqr_bounds,
            OUTLIER_GROUP_COLUMN,
            apply_target_to_train_only=bool(include_target_outlier),
        )
    else:
        iqr_bounds = fit_iqr_bounds(train_rows_encoded, outlier_columns, iqr_scale=iqr_scale)
        filtered, keep_mask = apply_iqr_bounds(
            encoded,
            iqr_bounds,
            apply_target_to_train_only=bool(include_target_outlier),
        )

    if filtered.empty:
        raise ValueError("No rows remain after train-fitted IQR filtering.")

    filtered = filtered.copy()
    filtered[PCA_INPUT_COLUMNS] = filtered[PCA_INPUT_COLUMNS].astype(np.float32)
    train_rows_filtered = filtered[filtered[INTERNAL_SPLIT_COLUMN].eq("train")]
    quantile_transformer = fit_quantile_transformer(train_rows_filtered)
    filtered.loc[:, PCA_INPUT_COLUMNS] = quantile_transformer.transform(filtered[PCA_INPUT_COLUMNS].astype(np.float32))

    metadata = {
        "leakage_safe": True,
        "categorical_fit_scope": "train_rows_only",
        "categorical_unknown_value": -1,
        "categorical_mappings": categorical_mappings,
        "outlier_scope": outlier_scope,
        "outlier_fit_scope": "train_rows_by_ufs" if outlier_scope in {"ufs", "ufs_group", "ufs_id"} else "train_rows_only",
        "outlier_apply_scope": (
            "same_ufs_rows_feature_columns_train_only_for_target_if_enabled"
            if outlier_scope in {"ufs", "ufs_group", "ufs_id"}
            else "all_rows_feature_columns_train_only_for_target_if_enabled"
        ),
        "outlier_columns_requested": list(OUTLIER_COLUMNS),
        "outlier_columns_used": outlier_columns,
        "target_outlier_filter_enabled": bool(include_target_outlier),
        "iqr_scale_requested": float(iqr_scale),
        "iqr_bounds": iqr_bounds,
        "iqr_bounds_by_ufs": group_iqr_bounds,
        "ufs_iqr_min_train_rows": int(min_ufs_outlier_rows),
        "ufs_iqr_fit_stats": outlier_group_fit_stats,
        "ufs_iqr_filter_stats": summarize_outlier_filter_by_group(encoded, keep_mask, OUTLIER_GROUP_COLUMN),
        "rows_before_split_filter": int(len(df)),
        "rows_after_split_filter": int(len(split_df)),
        "rows_after_iqr_filter": int(len(filtered)),
        "rows_removed_by_iqr_filter": int(len(split_df) - len(filtered)),
        "quantile_transformer_fit_scope": "train_rows_after_iqr_filter_only",
        "quantile_transformer_fit_rows": int(len(train_rows_filtered)),
        "quantile_transformer_n_quantiles": int(quantile_transformer.n_quantiles_),
    }
    return filtered, metadata


def repeat_and_trim(array, target_len):
    output = np.asarray(array)
    while output.shape[0] < target_len:
        output = np.concatenate([output, output], axis=0)
    return output[:target_len]


def minmax_normalize_2d(array, axis=None):
    array = np.asarray(array, dtype=np.float32)

    if axis is None:
        return (array - array.min()) / (array.max() - array.min() + 1e-8)

    min_value = array.min(axis=axis, keepdims=True)
    max_value = array.max(axis=axis, keepdims=True)
    return (array - min_value) / (max_value - min_value + 1e-8)


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
    return representative, minmax_normalize_2d(ema, axis=None)


def img_encoder(X, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, dpi=IMAGE_DPI):
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(X, linewidth=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.tight_layout(pad=0)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(height, width, 3)

    plt.close(fig)

    return image


def copy_canonical_dataset(source_dir, output_dir):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    required = ("X0.npy", "X1.npy", "y.npy")
    missing = [name for name in required if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Canonical dataset is missing required files: {}".format(", ".join(missing))
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in required:
        source_path = (source_dir / name).resolve()
        output_path = (output_dir / name).resolve()
        if source_path != output_path:
            shutil.copy2(source_path, output_path)

    optional_files = (
        "X2.npy",
        "categorical_feature_columns.json",
        "sample_ids.npy",
        "train_assignment.csv",
        "test_assignment.csv",
    )
    for name in optional_files:
        source_path = (source_dir / name).resolve()
        output_path = (output_dir / name).resolve()
        if source_path.exists() and source_path != output_path:
            shutil.copy2(source_path, output_path)

    X0 = np.load(output_dir / "X0.npy")
    X1 = np.load(output_dir / "X1.npy")
    y = np.load(output_dir / "y.npy")
    x2_path = output_dir / "X2.npy"
    X2 = np.load(x2_path) if x2_path.exists() else None
    summary_rows = [
        {
            "tag": TAG,
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "X0_shape": str(X0.shape),
            "X1_shape": str(X1.shape),
            "X2_shape": str(X2.shape) if X2 is not None else "",
            "y_shape": str(y.shape),
            "n_samples": int(len(y)),
            "mode": "canonical_copy",
        }
    ]
    save_csv(pd.DataFrame(summary_rows), output_dir / "dataset_summary.csv")
    save_json(
        {
            "tag": TAG,
            "mode": "canonical_copy",
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
        },
        output_dir / "preprocess_config.json",
    )
    return output_dir


def backup_and_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = dst.with_name(dst.name + f".bak_{timestamp}")
        shutil.copy2(dst, backup)
    shutil.copy2(src, dst)


def save_aligned_assignments(output_dir, sample_ids, sample_splits, y_values):
    rows = []
    for index, (sample_id, split, y_value) in enumerate(zip(sample_ids, sample_splits, y_values)):
        rows.append(
            {
                "sample_index": int(index),
                SAMPLE_ID_COLUMN: str(sample_id),
                "split": str(split),
                TARGET_COLUMN: float(np.asarray(y_value).reshape(-1)[0]),
            }
        )
    assignment_df = pd.DataFrame(rows)
    train_df = assignment_df[assignment_df["split"].eq("train")].copy()
    val_df = assignment_df[assignment_df["split"].eq("val")].copy()
    train_full_df = assignment_df[assignment_df["split"].isin(["train", "val"])].copy()
    train_full_df.loc[:, "split"] = "train_full"
    test_df = assignment_df[assignment_df["split"].eq("test")].copy()
    save_csv(assignment_df, output_dir / "all_assignment.csv")
    save_csv(train_df, output_dir / "train_assignment.csv")
    save_csv(val_df, output_dir / "val_assignment.csv")
    save_csv(train_full_df, output_dir / "train_full_assignment.csv")
    save_csv(test_df, output_dir / "test_assignment.csv")
    return assignment_df, train_df, val_df, train_full_df, test_df


def preprocess_dataset(
    csv_path,
    output_dir,
    split_source="auto",
    train_assignment=DEFAULT_REPRO_TRAIN_ASSIGNMENT,
    test_assignment=DEFAULT_REPRO_TEST_ASSIGNMENT,
    include_target_outlier=False,
    save_default_splits=False,
    categorical_mode=DEFAULT_CATEGORICAL_MODE,
    iqr_scale=1.5,
    outlier_scope="ufs",
    min_ufs_outlier_rows=50,
):
    np.random.seed(PREPROCESS_SEED)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw CSV file was not found: {csv_path}. "
            "Pass --csv-path to the actual sample_df.csv location."
        )

    df = read_raw_csv_robust(csv_path)
    df = df.copy()
    df[SAMPLE_ID_COLUMN] = df[SAMPLE_ID_COLUMN].astype(str)
    df = add_derived_columns(df)
    validate_required_columns(df)

    train_sample_ids, val_sample_ids, test_sample_ids, split_metadata = resolve_sample_split(
        df,
        split_source=split_source,
        train_assignment=train_assignment,
        test_assignment=test_assignment,
    )

    preprocessed_df, preprocessing_metadata = transform_common_dataframe_leakage_safe(
        df,
        train_sample_ids=train_sample_ids,
        val_sample_ids=val_sample_ids,
        test_sample_ids=test_sample_ids,
        include_target_outlier=include_target_outlier,
        iqr_scale=iqr_scale,
        outlier_scope=outlier_scope,
        min_ufs_outlier_rows=min_ufs_outlier_rows,
    )
    categorical_feature_columns = build_categorical_feature_columns(
        preprocessing_metadata.get("categorical_mappings", {})
    )
    categorical_mode = str(categorical_mode)
    if categorical_mode not in {"onehot_in_x0", "integer_in_x0", "normalized_integer_in_x0", "separate_x2"}:
        raise ValueError(
            "categorical_mode must be onehot_in_x0, integer_in_x0, normalized_integer_in_x0, or separate_x2."
        )
    x0_base_columns = (
        list(BASE_INPUT_COLUMNS)
        if categorical_mode in {"integer_in_x0", "normalized_integer_in_x0"}
        else SEPARATED_BASE_INPUT_COLUMNS
    )
    x0_input_columns = list(x0_base_columns)
    if categorical_mode == "onehot_in_x0":
        x0_input_columns = list(x0_input_columns) + list(categorical_feature_columns)
    preprocessing_metadata["categorical_mode"] = categorical_mode
    preprocessing_metadata["categorical_onehot_in_x0"] = bool(categorical_mode == "onehot_in_x0")
    preprocessing_metadata["categorical_normalized_integer_in_x0"] = bool(categorical_mode == "normalized_integer_in_x0")
    preprocessing_metadata["categorical_separated_as_x2"] = bool(categorical_mode == "separate_x2")
    preprocessing_metadata["x0_base_input_columns"] = list(x0_base_columns)
    preprocessing_metadata["x0_input_columns_before_pca"] = list(x0_input_columns)
    preprocessing_metadata["categorical_feature_columns"] = list(categorical_feature_columns)
    preprocessing_metadata["x2_feature_columns"] = list(categorical_feature_columns) if categorical_mode == "separate_x2" else []
    save_csv(preprocessed_df, output_dir / "preprocessed_df.csv")

    X0_all = []
    X1_all = []
    X2_all = []
    y_all = []
    sample_ids = []
    sample_splits = []
    log_rows = []

    requested_split_by_sample = {
        str(sample_id): "train" for sample_id in train_sample_ids
    }
    requested_split_by_sample.update({str(sample_id): "val" for sample_id in val_sample_ids})
    requested_split_by_sample.update({str(sample_id): "test" for sample_id in test_sample_ids})

    present_after_filter = set(preprocessed_df[SAMPLE_ID_COLUMN].astype(str).unique())
    for sample_id, split in sorted(requested_split_by_sample.items(), key=lambda item: item[0]):
        if sample_id not in present_after_filter:
            log_rows.append({SAMPLE_ID_COLUMN: sample_id, "split": split, "status": "skip_all_rows_after_iqr_filter"})

    for sample_key, group in tqdm(preprocessed_df.groupby(SAMPLE_ID_COLUMN, sort=True), desc="Preprocess {}".format(TAG)):
        sample_key = str(sample_key)
        split = str(group[INTERNAL_SPLIT_COLUMN].iloc[0])
        try:
            if len(group) <= 10:
                log_rows.append({SAMPLE_ID_COLUMN: sample_key, "split": split, "status": "skip_small", "n_rows": len(group)})
                continue

            y_value = aggregate_target_for_sample(group)
            if pd.isna(y_value):
                log_rows.append({SAMPLE_ID_COLUMN: sample_key, "split": split, "status": "skip_invalid_target", "n_rows": len(group)})
                continue
            base_input, X2 = build_base_input_for_x0(
                group,
                x0_base_columns=x0_base_columns,
                categorical_mappings=preprocessing_metadata.get("categorical_mappings", {}),
                categorical_mode=categorical_mode,
            )
            pca_input = repeat_and_trim(group[PCA_INPUT_COLUMNS].values, FIXED_SEQ_LEN)

            if np.unique(pca_input, axis=0).shape[0] <= N_COMPONENTS:
                log_rows.append({SAMPLE_ID_COLUMN: sample_key, "split": split, "status": "skip_pca", "n_rows": len(group)})
                continue

            pca = PCA(n_components=N_COMPONENTS)
            pca_features = pca.fit_transform(pca_input)
            full_input = np.concatenate([base_input, pca_features], axis=1)

            if np.unique(full_input, axis=0).shape[0] < N_CLUSTERS:
                log_rows.append({SAMPLE_ID_COLUMN: sample_key, "split": split, "status": "skip_kmeans", "n_rows": len(group)})
                continue

            kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=5, random_state=PREPROCESS_SEED)
            labels = kmeans.fit_predict(full_input)

            X0, X1_map = build_cluster_patterns(full_input, labels, out_len=OUT_LEN)
            if X0 is None or X1_map is None:
                log_rows.append({SAMPLE_ID_COLUMN: sample_key, "split": split, "status": "skip_pattern", "n_rows": len(group)})
                continue

            X1 = img_encoder(X1_map)
            X0_all.append(X0)
            X1_all.append(X1)
            if categorical_mode == "separate_x2":
                X2_all.append(X2)
            y_all.append(y_value)
            sample_ids.append(sample_key)
            sample_splits.append(split)
            target_stats = target_summary_for_sample(group)
            success_row = {SAMPLE_ID_COLUMN: sample_key, "split": split, "status": "success", "n_rows": len(group)}
            success_row.update(target_stats)
            log_rows.append(success_row)
        except Exception as exc:
            log_rows.append({SAMPLE_ID_COLUMN: sample_key, "split": split, "status": "error", "error": str(exc), "n_rows": len(group)})

    X0_array = np.asarray(X0_all, dtype=np.float32)
    X1_array = np.asarray(X1_all)
    X2_array = np.asarray(X2_all, dtype=np.float32) if categorical_mode == "separate_x2" else None
    y_array = np.asarray(y_all, dtype=np.float32).reshape(-1, 1)
    sample_ids_array = np.asarray(sample_ids)
    sample_splits_array = np.asarray(sample_splits)

    if len(y_array) == 0:
        raise RuntimeError("No valid samples were generated. Check group_log.csv and preprocessing thresholds.")

    assignment_df, train_assignment_df, val_assignment_df, train_full_assignment_df, test_assignment_df = save_aligned_assignments(
        output_dir,
        sample_ids=sample_ids_array,
        sample_splits=sample_splits_array,
        y_values=y_array,
    )

    if len(train_assignment_df) == 0 or len(test_assignment_df) == 0:
        raise RuntimeError(
            "The generated dataset has an empty train or test split. "
            f"train={len(train_assignment_df)}, test={len(test_assignment_df)}"
        )

    np.save(output_dir / "X0.npy", X0_array)
    np.save(output_dir / "X1.npy", X1_array)
    np.save(output_dir / "y.npy", y_array)
    np.save(output_dir / "sample_ids.npy", sample_ids_array)
    np.save(output_dir / "sample_splits.npy", sample_splits_array)
    save_json(categorical_feature_columns, output_dir / "categorical_feature_columns.json")
    if X2_array is not None:
        np.save(output_dir / "X2.npy", X2_array)

    save_csv(pd.DataFrame(log_rows), output_dir / "group_log.csv")

    if save_default_splits:
        backup_and_copy(output_dir / "train_assignment.csv", DEFAULT_REPRO_TRAIN_ASSIGNMENT)
        backup_and_copy(output_dir / "val_assignment.csv", DEFAULT_REPRO_VAL_ASSIGNMENT)
        backup_and_copy(output_dir / "test_assignment.csv", DEFAULT_REPRO_TEST_ASSIGNMENT)

    summary = {
        "tag": TAG,
        "mode": "raw_rebuild_leakage_safe",
        "csv_path": str(Path(csv_path)),
        "output_dir": str(output_dir),
        "X0_shape": str(X0_array.shape),
        "X1_shape": str(X1_array.shape),
        "X2_shape": str(X2_array.shape) if X2_array is not None else "",
        "y_shape": str(y_array.shape),
        "categorical_mode": categorical_mode,
        "n_samples": int(len(y_array)),
        "train_samples": int(len(train_assignment_df)),
        "val_samples": int(len(val_assignment_df)),
        "train_full_samples": int(len(train_full_assignment_df)),
        "test_samples": int(len(test_assignment_df)),
        "split_source": split_metadata.get("source"),
        "train_assignment_output": str(output_dir / "train_assignment.csv"),
        "val_assignment_output": str(output_dir / "val_assignment.csv"),
        "train_full_assignment_output": str(output_dir / "train_full_assignment.csv"),
        "test_assignment_output": str(output_dir / "test_assignment.csv"),
    }
    save_csv(pd.DataFrame([summary]), output_dir / "dataset_summary.csv")

    save_json(
        {
            "tag": TAG,
            "mode": "raw_rebuild_leakage_safe",
            "n_components": N_COMPONENTS,
            "n_clusters": N_CLUSTERS,
            "csv_path": str(Path(csv_path)),
            "output_dir": str(output_dir),
            "split_metadata": split_metadata,
            "preprocessing_metadata": preprocessing_metadata,
            "assignment_outputs": {
                "all": str(output_dir / "all_assignment.csv"),
                "train": str(output_dir / "train_assignment.csv"),
                "val": str(output_dir / "val_assignment.csv"),
                "train_full": str(output_dir / "train_full_assignment.csv"),
                "test": str(output_dir / "test_assignment.csv"),
                "also_saved_to_default_paths": bool(save_default_splits),
                "default_train_assignment": str(DEFAULT_REPRO_TRAIN_ASSIGNMENT),
                "default_val_assignment": str(DEFAULT_REPRO_VAL_ASSIGNMENT),
                "default_test_assignment": str(DEFAULT_REPRO_TEST_ASSIGNMENT),
            },
        },
        output_dir / "preprocess_config.json",
    )
    return output_dir
