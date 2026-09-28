import pandas as pd
import pickle
from pathlib import Path

from feature_engineering import engineer_features, align_columns, load_stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_TRANSACTION = DATA_DIR / "train_transaction.txt"
TRAIN_IDENTITY = DATA_DIR / "train_identity.txt"
STATS_PATH = DATA_DIR / "training_stats.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "catboost_feature_columns.pkl"

# --------------------------------------------------
# Load a small sample of the raw training data
# --------------------------------------------------

print("Loading raw data...")

transaction = pd.read_csv(TRAIN_TRANSACTION, nrows=1000)
identity = pd.read_csv(TRAIN_IDENTITY)

sample = transaction.merge(
    identity,
    on="TransactionID",
    how="left"
)

print("Raw sample shape:", sample.shape)


# --------------------------------------------------
# Load saved training statistics
# --------------------------------------------------

print("Loading training statistics...")

stats = load_stats(STATS_PATH)


# --------------------------------------------------
# Engineer features
# --------------------------------------------------

print("Engineering features...")

features = engineer_features(sample, stats)


# --------------------------------------------------
# Load exact model feature schema
# --------------------------------------------------

print("Loading feature schema...")

with open(FEATURE_COLUMNS_PATH, "rb") as f:
    feature_columns = pickle.load(f)


# --------------------------------------------------
# Align to exact model schema
# --------------------------------------------------

features = align_columns(features, feature_columns)


# --------------------------------------------------
# Verification
# --------------------------------------------------

print("\n========== VERIFICATION ==========")

print("Final shape:", features.shape)
print("Expected columns:", len(feature_columns))
print("Actual columns:", len(features.columns))

print(
    "Column order correct:",
    features.columns.tolist() == feature_columns
)

print(
    "Missing values:",
    features.isna().sum().sum()
)

print("\nFirst 10 columns:")
print(features.columns[:10].tolist())

print("\nLast 10 columns:")
print(features.columns[-10:].tolist())

print("\n==================================")