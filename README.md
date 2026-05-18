# v_github

This folder contains the final execution pipeline derived from v6. It keeps only the files needed for preprocessing, training, train-based correction, and test output generation.

## Files

- `config.py`: paths, column names, preprocessing settings, and fixed ensemble member settings
- `preprocess_core.py`: CSV-to-array preprocessing engine
- `preprocess.py`: preprocessing entrypoint
- `model.py`: final X0/X1 model definition
- `train.py`: fixed ensemble training, train-based correction, and test prediction output
- `run_pipeline.py`: full pipeline entrypoint
- `requirements.txt`: Python package requirements

## Input

The default input file is `sample_df.csv` in the project root:

```text
D:\projects\UFS-githun\sample_df.csv
```

To use another CSV path, pass `--csv-path` or set the `UFS_SAMPLE_CSV` environment variable.

## Run

Run the full pipeline:

```powershell
cd D:\projects\UFS-githun\v_github
& "C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python36_64\python.exe" run_pipeline.py
```

Run with an explicit CSV path:

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python36_64\python.exe" run_pipeline.py --csv-path D:\projects\UFS-githun\sample_df.csv
```

Reuse existing preprocessing artifacts and run training only:

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python36_64\python.exe" run_pipeline.py --skip-preprocess
```

Rebuild preprocessing artifacts:

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python36_64\python.exe" run_pipeline.py --force-preprocess
```

## Output

Outputs are written under:

```text
artifacts/v_github/
```

Main output folders:

- `data/PC3_CL6/`: preprocessed arrays and split assignment files
- `outputs/final_ensemble/ensemble_run_YYYYMMDD_HHMMSS/`: member outputs, ensemble manifest, and test prediction CSV files
