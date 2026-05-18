import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(command):
    print("\n> {}".format(" ".join(command)))
    subprocess.check_call(command, cwd=str(SCRIPT_DIR))


def main():
    parser = argparse.ArgumentParser(description="Run preprocessing, training, correction, and test output generation.")
    parser.add_argument("--csv-path", default="", help="Input CSV path passed to preprocess.py.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Use existing dataset artifacts.")
    parser.add_argument("--force-preprocess", action="store_true", help="Rebuild dataset artifacts before training.")
    parser.add_argument("--verbose", type=int, default=0, help="Keras fit verbosity.")
    args = parser.parse_args()

    if not args.skip_preprocess:
        preprocess_command = [sys.executable, str(SCRIPT_DIR / "preprocess.py")]
        if args.csv_path:
            preprocess_command.extend(["--csv-path", str(args.csv_path)])
        if args.force_preprocess:
            preprocess_command.append("--force")
        run_step(preprocess_command)

    run_step([sys.executable, str(SCRIPT_DIR / "train.py"), "--verbose", str(int(args.verbose))])


if __name__ == "__main__":
    main()
