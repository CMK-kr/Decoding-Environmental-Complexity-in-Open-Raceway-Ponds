# GitHub Release Pipeline

This folder contains a simplified preprocessing and training pipeline for the microalgae prediction model.

## Included files

- `config.py`: fixed feature definitions, best PCA/K settings, and best model hyperparameters
- `preprocessing.py`: converts the raw CSV into `X0.npy`, `X1.npy`, and `y.npy`
- `model.py`: multimodal model definition
- `train.py`: single train/validation/test split training script
- `run_pipeline.py`: runs preprocessing and training end-to-end

## Fixed settings

- PCA components: `12`
- K-means clusters: `17`
- Best model hyperparameters:
  - `filters1=384`
  - `filters2=544`
  - `filters3=416`
  - `filters4=416`
  - `filters5=160`
  - `filters6=160`
  - `dropout=0.23`
  - `lr=0.0003658726548219748`

## Usage

Run preprocessing only:

```bash
python -m github_release.preprocessing --csv-file "path/to/sample_df.csv" --output-root "path/to/output"
```

Run training only:

```bash
python -m github_release.train --dataset-dir "path/to/output/PC12_CL17" --output-dir "path/to/train_output"
```

Run the full pipeline:

```bash
python -m github_release.run_pipeline --csv-file "path/to/sample_df.csv" --work-dir "path/to/work_dir"
```
