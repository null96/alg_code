"""
AutoML Pipeline Configuration
"""
import os

# Data paths (override via --data-dir CLI arg)
DATA_DIR = os.environ.get("LOCAL_DATA_DIR", r"F:\Automl_test_data\af_class")
TRAIN_FILE = "anzhen2022_2024_move_cut_features_train.csv"
TEST_FILE = "anzhen2022_2024_move_cut_features_test.csv"

# Target
TARGET_COL = "af_annotation_label"

# Non-feature columns to drop
DROP_COLS = ["file_name", "source_record", "dataset_split"]

# Output
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
MODEL_FILE = os.path.join(OUTPUT_DIR, "best_model.pkl")
METRICS_FILE = os.path.join(OUTPUT_DIR, "metrics.json")
PREPROCESSOR_FILE = os.path.join(OUTPUT_DIR, "preprocessor.pkl")

# Random seed
SEED = 42

# Optuna settings
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = 1800  # 30 minutes
