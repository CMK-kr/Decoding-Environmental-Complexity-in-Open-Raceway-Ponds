import argparse
from pathlib import Path

import pandas as pd

from config import DATASET_DIR, RAW_DATA_CSV_PATH
from preprocess_core import preprocess_dataset


def dataset_exists(dataset_dir):
    dataset_dir = Path(dataset_dir)
    return all((dataset_dir / name).exists() for name in ("X0.npy", "X1.npy", "y.npy", "sample_ids.npy", "train_full_assignment.csv", "test_assignment.csv"))


def summarize_dataset(dataset_dir):
    summary_path = Path(dataset_dir) / "dataset_summary.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return
    row = summary.iloc[0].to_dict()
    print(
        "Dataset samples: total={n_samples}, train={train_samples}, val={val_samples}, "
        "train_full={train_full_samples}, test={test_samples}".format(**row)
    )


def main():
    parser = argparse.ArgumentParser(description="Build preprocessing artifacts for the final pipeline.")
    parser.add_argument("--csv-path", default=str(RAW_DATA_CSV_PATH), help="Input CSV path. Defaults to UFS_SAMPLE_CSV or ../sample_df.csv.")
    parser.add_argument("--force", action="store_true", help="Rebuild artifacts even when the dataset already exists.")
    args = parser.parse_args()

    if dataset_exists(DATASET_DIR) and not args.force:
        print("Dataset already exists: {}".format(DATASET_DIR))
        summarize_dataset(DATASET_DIR)
        return

    output_dir = preprocess_dataset(
        csv_path=Path(args.csv_path),
        output_dir=DATASET_DIR,
        split_source="ufs_target_tvt",
        include_target_outlier=False,
        save_default_splits=False,
        categorical_mode="normalized_integer_in_x0",
        iqr_scale=1.55,
        outlier_scope="global",
    )
    print("Preprocessing finished: {}".format(output_dir))
    print("Split policy: UFS + target-bin train/val/test split.")
    summarize_dataset(output_dir)


if __name__ == "__main__":
    main()
