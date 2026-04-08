import argparse
import json
import os

from github_release.preprocessing import preprocess_dataset
from github_release.train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed preprocessing and training pipeline.")
    parser.add_argument("--csv-file", required=True, help="Path to the raw CSV file.")
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Root directory where dataset and training artifacts will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = os.path.join(args.work_dir, "dataset")
    training_root = os.path.join(args.work_dir, "training")

    dataset_dir = preprocess_dataset(args.csv_file, dataset_root)
    result = train_model(dataset_dir, training_root)

    print("[DONE] full pipeline finished")
    print(json.dumps({"dataset_dir": dataset_dir, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
