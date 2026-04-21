# Final Release Scripts

This directory contains the cleaned final pipeline for the best-performing setting.
You can publish this directory alone. The default raw CSV path is `./sample_df.csv`.

- Best combination: `PC3_CL6`
- Holdout split: `StratifiedShuffleSplit`, `test_size=0.2`, `seed=46`
- Best hyperparameters: fixed in `config.py`
- Final training target: `X0 + X1` only

## Files

- `config.py`
  - Final experiment constants, best `PC*_CL*` combination, and tuned hyperparameters.
- `preprocess_dataset.py`
  - Generates `X0.npy`, `X1.npy`, `y.npy`, and `sample_ids.npy` from `sample_df.csv`.
- `model.py`
  - Final multimodal model definition and custom loss function.
- `train_model.py`
  - Trains the final `X0 + X1` model with a single holdout split.

## Usage

### 1. Preprocess the dataset

```bash
python preprocess_dataset.py
```

Default input:

- `./sample_df.csv`

Default output:

- `./artifacts/PC3_CL6`

### 2. Train the model

Run the final multimodal model:

```bash
python train_model.py --dataset-dir ./artifacts/PC3_CL6
```

Default training output:

- `./artifacts/training/x0_x1_single_holdout`

## Generated Outputs

### Preprocessing

- `X0.npy`
- `X1.npy`
- `y.npy`
- `sample_ids.npy`
- `group_log.csv`
- `dataset_summary.csv`

### Training

- `train_metrics.csv`
- `test_metrics.csv`
- `split_metrics.csv`
- `train_predictions.csv`
- `test_predictions.csv`
- `keras_history.csv`
- `train_assignment.csv`
- `test_assignment.csv`
- `run_config.json`
- `training_summary.csv`
