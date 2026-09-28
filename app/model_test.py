import os
import joblib

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import Booster


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


print("Loading CatBoost...")
catboost_model = CatBoostClassifier()
catboost_model.load_model(
    os.path.join(MODEL_DIR, "catboost_final.cbm")
)

print("Loading XGBoost...")
xgb_models = []

for seed in [123, 2024, 777]:
    model = XGBClassifier()
    model.load_model(
        os.path.join(MODEL_DIR, f"xgb_seed_{seed}.json")
    )
    xgb_models.append(model)

print("Loading LightGBM...")
lgb_models = []

for seed in [123, 2024, 777]:
    model = Booster(
        model_file=os.path.join(
            MODEL_DIR,
            f"lgb_seed_{seed}.txt"
        )
    )
    lgb_models.append(model)

print("Loading preprocessors...")

xgb_preprocessor = joblib.load(
    os.path.join(MODEL_DIR, "xgb_preprocessor.pkl")
)

lgb_preprocessor = joblib.load(
    os.path.join(MODEL_DIR, "lgb_preprocessor.pkl")
)

print("Loading meta-model...")

meta_model = joblib.load(
    os.path.join(MODEL_DIR, "final_meta_model.pkl")
)

meta_features = joblib.load(
    os.path.join(MODEL_DIR, "meta_feature_columns.pkl")
)

print("\nEverything loaded successfully.")

print("CatBoost:", type(catboost_model))
print("XGBoost:", len(xgb_models), "models")
print("LightGBM:", len(lgb_models), "models")
print("Meta model:", type(meta_model))
print("Meta features:", meta_features)