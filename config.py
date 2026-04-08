from dataclasses import asdict, dataclass


SAMPLE_ID_COL = "sampleID"
TARGET_COL = "AFDW..g."

STR_FEATURES = ["SiteID", "PondID", "StrainID"]
FEATURES = [
    "pH",
    "GlobalLightEnergy(W.m2)",
    "Sal (g.L)",
    "Cond (mS.cm)",
    "Temp (C)",
    "DO (mg.L)",
    "DO (%sat)",
]
DATE_FEATURES = ["Year", "Month", "Day", "Hour"]
STATIC_FEATURES = [
    "time.between.harvests.days",
    "Harvest.",
    "Depth.cm",
    "NO3.mg.L",
    "P.mg.L",
    "N.P.ratio",
]

PCA_INPUT_COLS = FEATURES + DATE_FEATURES + STATIC_FEATURES
BASE_INPUT_COLS = PCA_INPUT_COLS + STR_FEATURES
OUTLIER_COLS = FEATURES + [TARGET_COL] + STATIC_FEATURES + DATE_FEATURES

BEST_PC_COMPONENTS = 12
BEST_K_CLUSTERS = 17
BEST_MODEL_PARAMS = {
    "filters1": 384,
    "filters2": 544,
    "filters3": 416,
    "filters4": 416,
    "filters5": 160,
    "filters6": 160,
    "dropout": 0.23,
    "lr": 0.0003658726548219748,
}


@dataclass(frozen=True)
class PreprocessingConfig:
    fixed_seq_len: int = 6080
    out_len: int = 112
    ema_alpha: float = 0.95
    n_components: int = BEST_PC_COMPONENTS
    n_clusters: int = BEST_K_CLUSTERS
    min_group_size: int = 10
    kmeans_n_init: int = 5
    random_seed: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1000
    batch_size: int = 64
    random_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    early_stopping_patience: int = 30
    reduce_lr_patience: int = 15
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    n_bins: int = 5


def preprocessing_config_dict():
    return asdict(PreprocessingConfig())


def training_config_dict():
    return asdict(TrainingConfig())
