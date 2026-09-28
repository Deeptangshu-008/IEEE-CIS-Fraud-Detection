from pathlib import Path
import pickle
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

from feature_engineering import engineer_features


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# LOAD ARTIFACTS
# ============================================================

print("Loading models and preprocessors...")


# ---------- CatBoost ----------
catboost_model = CatBoostClassifier()
catboost_model.load_model(
    str(MODELS_DIR / "catboost_final.cbm")
)

with open(MODELS_DIR / "catboost_feature_columns.pkl", "rb") as f:
    catboost_feature_columns = pickle.load(f)

with open(MODELS_DIR / "catboost_categorical_columns.pkl", "rb") as f:
    catboost_categorical_columns = pickle.load(f)


# ---------- XGBoost ----------
xgb_models = []

for seed in [123, 2024, 777]:
    model = XGBClassifier()
    model.load_model(
        str(MODELS_DIR / f"xgb_seed_{seed}.json")
    )
    xgb_models.append(model)

import joblib

xgb_preprocessor = joblib.load(MODELS_DIR / "xgb_preprocessor.pkl")

# ---------- LightGBM ----------
lgb_models = []

for seed in [123, 2024, 777]:
    model = lgb.Booster(
        model_file=str(MODELS_DIR / f"lgb_seed_{seed}.txt")
    )
    lgb_models.append(model)


lgb_preprocessor = joblib.load(MODELS_DIR / "lgb_preprocessor.pkl")

# ---------- Meta model ----------
meta_model = joblib.load(MODELS_DIR / "final_meta_model.pkl")

with open(MODELS_DIR / "meta_feature_columns.pkl", "rb") as f:
    meta_feature_columns = pickle.load(f)


# ---------- Training statistics ----------
with open(MODELS_DIR / "training_stats.pkl", "rb") as f:
    training_stats = pickle.load(f)
print("All artifacts loaded successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_fraud(transaction_df):
    """
    Generate fraud probabilities for raw transaction data.

    Parameters
    ----------
    transaction_df : pandas.DataFrame
        Raw transaction dataframe.

    Returns
    -------
    numpy.ndarray
        Final fraud probabilities.
    """

    print("\nEngineering features...")

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    features = engineer_features(
        transaction_df.copy(),
        training_stats
    )

    # Make absolutely sure CatBoost receives
    # the exact training column order.
    features = features[
        catboost_feature_columns
    ]

    print(f"Engineered shape: {features.shape}")


    # ========================================================
    # CATBOOST
    # ========================================================

    print("Running CatBoost...")

    catboost_input = features[catboost_feature_columns].copy()

    for col in catboost_categorical_columns:
        catboost_input[col] = (
            catboost_input[col]
            .fillna("Missing")
            .astype(str)
        )

    catboost_pred = catboost_model.predict_proba(catboost_input)[:, 1]


    # ========================================================
    # XGBOOST
    # ========================================================

    print("Running XGBoost...")

    xgb_input = xgb_preprocessor.transform(features)

    xgb_predictions = []

    for model in xgb_models:

        pred = model.predict_proba(
            xgb_input
        )[:, 1]

        xgb_predictions.append(pred)

    xgb_pred = np.mean(
        xgb_predictions,
        axis=0
    )


    # ========================================================
    # LIGHTGBM
    # ========================================================

    print("Running LightGBM...")

    lgb_input = lgb_preprocessor.transform(features)

    lgb_predictions = []

    for model in lgb_models:

        pred = model.predict(
            lgb_input
        )

        lgb_predictions.append(pred)

    lgb_pred = np.mean(
        lgb_predictions,
        axis=0
    )


    # ========================================================
    # META FEATURES
    # ========================================================

    meta_features = pd.DataFrame({

        "catboost": catboost_pred,

        "lightgbm": lgb_pred,

        "xgboost": xgb_pred

    })

    meta_features = meta_features[
        meta_feature_columns
    ]


    # ========================================================
    # FINAL STACKING MODEL
    # ========================================================

    print("Running stacking meta-model...")

    final_pred = meta_model.predict_proba(
        meta_features
    )[:, 1]


    print("Prediction complete.")

    return final_pred

if __name__ == "__main__":

    print("\nLoading test data...")

    test_transaction = pd.read_csv(DATA_DIR / "test_transaction.txt")
    test_identity = pd.read_csv(DATA_DIR / "test_identity.csv")

    # Normalize identity column names
    test_identity.columns = test_identity.columns.str.replace(
        "-", "_", regex=False
    )

    test_df = test_transaction.merge(
        test_identity,
        on="TransactionID",
        how="left"
    )

    # Test only 100 rows first
    sample = test_df.head(100)

    predictions = predict_fraud(sample)

    print("\n========== INFERENCE TEST ==========")

    print("Number of predictions:", len(predictions))

    print("First 10 predictions:")
    print(predictions[:10])

    print("Minimum probability:", predictions.min())
    print("Maximum probability:", predictions.max())

    print("====================================")