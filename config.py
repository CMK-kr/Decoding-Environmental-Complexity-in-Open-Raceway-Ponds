import os
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent

MODE_NAME = "v_github"
ARTIFACT_ROOT = CURRENT_DIR / "artifacts" / MODE_NAME
DATA_ROOT = ARTIFACT_ROOT / "data"
OUTPUT_ROOT = ARTIFACT_ROOT / "outputs"

N_COMPONENTS = 3
N_CLUSTERS = 6
TAG = "PC{}_CL{}".format(N_COMPONENTS, N_CLUSTERS)

DATASET_DIR = DATA_ROOT / TAG
RAW_DATA_CSV_PATH = Path(os.environ.get("UFS_SAMPLE_CSV", PROJECT_DIR / "sample_df.csv"))

DEFAULT_REPRO_TRAIN_ASSIGNMENT = DATASET_DIR / "train_assignment.csv"
DEFAULT_REPRO_VAL_ASSIGNMENT = DATASET_DIR / "val_assignment.csv"
DEFAULT_REPRO_TRAIN_FULL_ASSIGNMENT = DATASET_DIR / "train_full_assignment.csv"
DEFAULT_REPRO_TEST_ASSIGNMENT = DATASET_DIR / "test_assignment.csv"
DEFAULT_REPRO_OUTPUT_DIR = OUTPUT_ROOT / "final_ensemble"

SAMPLE_ID_COLUMN = "sampleID"
TARGET_COLUMN = "AFDW..g.L."

CATEGORICAL_COLUMNS = ("SiteID", "PondID", "StrainID")
SENSOR_FEATURES = (
    "pH",
    "GlobalLightEnergy(W.m2)",
    "Sal (g.L)",
    "Cond (mS.cm)",
    "Temp (C)",
    "DO (mg.L)",
    "DO (%sat)",
)
DATE_FEATURES = ("Year", "Month", "Day", "Hour")
STATIC_FEATURES = (
    "time.between.harvests.days",
    "Harvest.",
    "Depth.cm",
    "NO3.mg.L",
    "P.mg.L",
    "N.P.ratio",
)

PCA_INPUT_COLUMNS = list(SENSOR_FEATURES + DATE_FEATURES + STATIC_FEATURES)
BASE_INPUT_COLUMNS = PCA_INPUT_COLUMNS + list(CATEGORICAL_COLUMNS)

FIXED_SEQ_LEN = 6080
OUT_LEN = 112
IMAGE_WIDTH = 168
IMAGE_HEIGHT = 112
IMAGE_DPI = 300

TEST_SIZE = 0.20
VAL_SIZE = 0.16
SPLIT_SEED = 46
PREPROCESS_SEED = SPLIT_SEED
VAL_SPLIT_SEED = 146
N_BINS = 5
UFS_TARGET_SPLIT_SEED = None

COVARIATE_SPLIT_NUMERIC_COLUMNS = list(PCA_INPUT_COLUMNS)
COVARIATE_SPLIT_CATEGORICAL_COLUMNS = list(CATEGORICAL_COLUMNS)
COVARIATE_SPLIT_PCA_COMPONENTS = 3
COVARIATE_SPLIT_CLUSTERS = 6
COVARIATE_SPLIT_CANDIDATE_SEED_COUNT = 80
SPLIT_CANDIDATE_SEED_COUNT = COVARIATE_SPLIT_CANDIDATE_SEED_COUNT
UFS_PRIMARY_SPLIT_WEIGHT = 10.0

DISTRIBUTION_WEIGHT_PCA_COMPONENTS = 3
DISTRIBUTION_WEIGHT_CLUSTERS = 6
DISTRIBUTION_WEIGHT_SEED = 46

OUTLIER_FEATURE_COLUMNS = list(SENSOR_FEATURES) + list(STATIC_FEATURES) + list(DATE_FEATURES)
OUTLIER_COLUMNS = list(OUTLIER_FEATURE_COLUMNS)

MAX_TRAINABLE_PARAMS = 4_050_000
BEST_SINGLE_MEMBER_ID = "member_01"
ENSEMBLE_MANIFEST_NAME = "ensemble_manifest.json"

MODEL_BASE_PARAMS = {
    "x0_dense1": 219,
    "x0_dense2": 258,
    "x0_dense3": 236,
    "x0_dense4": 136,
    "x0_conv1": 104,
    "x0_conv2": 144,
    "x0_conv3": 117,
    "x0_conv4": 153,
    "x0_attn_hidden": 15,
    "x0_attn_channels": 126,
    "x0_proj": 134,
    "x1_conv1": 162,
    "x1_conv2": 211,
    "x1_conv3": 146,
    "x1_conv4": 94,
    "x1_attn_hidden": 17,
    "x1_attn_channels": 107,
    "x1_proj": 133,
}
BEST_PARAMS = dict(MODEL_BASE_PARAMS, dropout=0.12670048866458633, lr=0.0005150238281333329)

FINAL_ENSEMBLE_MEMBERS = [
    {
        "member_id": "member_01",
        "seed": 48,
        "epochs": 70,
        "params": dict(BEST_PARAMS, loss_type="mae"),
        "sample_weight_mode": "covariate_cluster",
        "sample_weight_strength": 0.14019649914873092,
        "sample_weight_max": 2.2873194810411013,
        "prediction_correction": "linear",
        "correction_strength": 0.4378183238617759,
    },
    {
        "member_id": "member_02",
        "seed": 21,
        "epochs": 56,
        "params": dict(BEST_PARAMS, loss_type="mae"),
        "sample_weight_mode": "covariate_cluster",
        "sample_weight_strength": 0.14019649914873092,
        "sample_weight_max": 2.2873194810411013,
        "prediction_correction": "linear",
        "correction_strength": 0.4378183238617759,
    },
    {
        "member_id": "member_03",
        "seed": 49,
        "epochs": 42,
        "params": dict(
            MODEL_BASE_PARAMS,
            dropout=0.10179732231348944,
            lr=0.0005623037761667329,
            loss_type="mse_mae",
            loss_mse_weight=7.752314570374875,
            loss_mae_weight=1.0,
        ),
        "sample_weight_mode": "target_covariate",
        "sample_weight_strength": 0.20134853261900493,
        "sample_weight_max": 2.236137594217678,
        "prediction_correction": "none",
        "correction_strength": 0.0,
    },
    {
        "member_id": "member_04",
        "seed": 50,
        "epochs": 52,
        "params": dict(
            MODEL_BASE_PARAMS,
            dropout=0.1286475030388328,
            lr=0.0005074109487567159,
            loss_type="mae",
        ),
        "sample_weight_mode": "covariate_cluster",
        "sample_weight_strength": 0.13392521695806203,
        "sample_weight_max": 2.2774341074801696,
        "prediction_correction": "none",
        "correction_strength": 0.0,
    },
    {
        "member_id": "member_05",
        "seed": 73,
        "epochs": 47,
        "params": dict(
            MODEL_BASE_PARAMS,
            dropout=0.10179732231348944,
            lr=0.0005623037761667329,
            loss_type="mse_mae",
            loss_mse_weight=7.752314570374875,
            loss_mae_weight=1.0,
        ),
        "sample_weight_mode": "target_covariate",
        "sample_weight_strength": 0.20134853261900493,
        "sample_weight_max": 2.236137594217678,
        "prediction_correction": "none",
        "correction_strength": 0.0,
    },
]
