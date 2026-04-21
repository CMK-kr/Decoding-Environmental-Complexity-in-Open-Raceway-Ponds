from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent

DATA_CSV_PATH = CURRENT_DIR / "sample_df.csv"
ARTIFACT_ROOT = CURRENT_DIR / "artifacts"
DATASET_DIR = ARTIFACT_ROOT / "PC3_CL6"
TRAINING_ROOT = ARTIFACT_ROOT / "training"

N_COMPONENTS = 3
N_CLUSTERS = 6
TAG = "PC{}_CL{}".format(N_COMPONENTS, N_CLUSTERS)

FIXED_SEQ_LEN = 6080
OUT_LEN = 112
IMAGE_WIDTH = 168
IMAGE_HEIGHT = 112
IMAGE_DPI = 112

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
OUTLIER_COLUMNS = list(SENSOR_FEATURES) + [TARGET_COLUMN] + list(STATIC_FEATURES) + list(DATE_FEATURES)

TEST_SIZE = 0.2
SPLIT_SEED = 46
N_BINS = 5
EPOCHS = 1000
BATCH_SIZE = 64

BEST_PARAMS = {
    "x0_dense1": 501,
    "x0_dense2": 594,
    "x0_dense3": 550,
    "x0_dense4": 317,
    "x0_conv1": 255,
    "x0_conv2": 353,
    "x0_conv3": 275,
    "x0_conv4": 356,
    "x0_attn_hidden": 34,
    "x0_attn_channels": 315,
    "x0_proj": 336,
    "x1_conv1": 371,
    "x1_conv2": 469,
    "x1_conv3": 317,
    "x1_conv4": 215,
    "x1_attn_hidden": 39,
    "x1_attn_channels": 260,
    "x1_proj": 307,
    "dropout": 0.07874573569726473,
    "lr": 0.0006136231584454102,
}
