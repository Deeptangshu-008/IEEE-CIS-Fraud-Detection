"""
Deployment-safe feature engineering for the fraud-detection model.

Design
------
The pipeline is split into two kinds of features:

1. "Base" features (`build_base_features`) — deterministic, row-level
   transformations that need nothing but the incoming transaction itself
   (time parts, log amount, cents, email parsing, uuid interaction keys,
   missingness flags, browser/OS, M-column mapping, ...). These are safe
   to compute on a single incoming transaction.

2. "Training-learned statistics" (`fit_training_stats` /
   `apply_training_stats`) — anything that used to be a `.groupby(...)`
   over the incoming dataframe (the card1 mean/std z-score, and the
   uuid2-based group means). These CANNOT be recomputed from a single
   inference request — a batch of one transaction has no distribution to
   group over, and even a large batch would leak inference-time data into
   what should be training-time statistics. Instead:
       - `fit_training_stats(train_df)` is run ONCE, offline, on the raw
         training data, and returns a dict of lookup tables.
       - That dict is pickled and shipped alongside the trained model.
       - `apply_training_stats(df, stats)` only ever does `.map()` /
         MultiIndex lookups against those saved tables — it never calls
         `.groupby()` on inference-time data.

`engineer_features(df, stats)` is the single function `inference.py`
should import and call per request: base features + saved stats applied,
in that order (base features must exist before the stats lookups can key
into them, e.g. uuid2 must be built before its group stats can be joined).

`align_columns(df, feature_columns)` enforces the exact saved column
order the model was trained on.

Nothing in this module executes at import time. The `if __name__ ==
"__main__":` block at the bottom is the OFFLINE fitting workflow (reads
the full training CSVs, fits the stats, pickles them) — it is not, and
must not be, invoked by the deployed app.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

from functools import partial

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Constants used by the feature functions
# ----------------------------------------------------------------------

COL_ID = "TransactionID"
COL_DT = "TransactionDT"
COL_AMOUNT = "TransactionAmt"
COL_TARGET = "isFraud"

INTERACTIONS = [
    "card1__addr1", "card1__card5", "card2__dist1", "card2__id_20", "card5__P_emaildomain",
    "card1__card2__card3__card5", "card1__card2__card3__card5__addr1__addr2",
]

RENAME = {
    "card1__card2__card3__card5": "uuid2",
    "card1__card2__card3__card5__addr1__addr2": "uuid3",
}

# NOTE: carried over from the original notebook, where it was declared but
# never actually applied (no frequency-encoding step used it). Left here
# for parity with the saved schema investigation; wire it up only if the
# 476-column schema turns out to need frequency-encoded versions of these.
ENCODE_FREQ_COLS = ["card1__addr1", "card1__card5", "uuid3"]

# (group_cols, target_col, agg) — these become training-learned lookups,
# not live groupbys, at inference time.
GROUP_AGGS = [
    (["uuid2", "ProductCD"], COL_AMOUNT, "mean"),
    (["uuid2", "addr1"], "dist2", "mean"),
]

NA_FLAG_COLS = ["D2", "D3", "D5", "D6", "D7", "D13", "D14", "M4", "M5", "id_16"]

EMAIL_MAP = {
    "gmail": "google", "att.net": "att", "twc.com": "spectrum", "scranton.edu": "other", "optonline.net": "other",
    "hotmail.co.uk": "microsoft", "comcast.net": "other", "yahoo.com.mx": "yahoo", "yahoo.fr": "yahoo",
    "yahoo.es": "yahoo", "charter.net": "spectrum", "live.com": "microsoft", "aim.com": "aol",
    "hotmail.de": "microsoft", "centurylink.net": "centurylink", "gmail.com": "google", "me.com": "apple",
    "earthlink.net": "other", "gmx.de": "other", "web.de": "other", "cfl.rr.com": "other", "hotmail.com": "microsoft",
    "protonmail.com": "other", "hotmail.fr": "microsoft", "windstream.net": "other", "outlook.es": "microsoft",
    "yahoo.co.jp": "yahoo", "yahoo.de": "yahoo", "servicios-ta.com": "other", "netzero.net": "other",
    "suddenlink.net": "other", "roadrunner.com": "other", "sc.rr.com": "other", "live.fr": "microsoft",
    "verizon.net": "yahoo", "msn.com": "microsoft", "q.com": "centurylink", "prodigy.net.mx": "att",
    "frontier.com": "yahoo", "anonymous.com": "other", "rocketmail.com": "yahoo", "sbcglobal.net": "att",
    "frontiernet.net": "yahoo", "ymail.com": "yahoo", "outlook.com": "microsoft", "mail.com": "other",
    "bellsouth.net": "other", "embarqmail.com": "centurylink", "cableone.net": "other", "hotmail.es": "microsoft",
    "mac.com": "apple", "yahoo.co.uk": "yahoo", "netzero.com": "other", "yahoo.com": "yahoo", "live.com.mx": "microsoft",
    "ptd.net": "other", "cox.net": "other", "aol.com": "aol", "juno.com": "other", "icloud.com": "apple",
}

BROWSERS = ["chrome", "safari", "ie ", "firefox", "edge", "samsung", "opera"]
OS_NAMES = ["windows", "ios", "mac", "android", "linux"]
M_BINARY_COLS = ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]


# ----------------------------------------------------------------------
# Small helper functions (unchanged from the notebook)
# ----------------------------------------------------------------------

def feature_cents(df: pd.DataFrame) -> pd.Series:
    return (np.modf(df["TransactionAmt"])[0] * 1000).astype(np.uint16)


def feature_ProductCD_W_cents(df: pd.DataFrame) -> pd.Series:
    a = (df["Cents"] == 0) | (df["Cents"] == 500) | (df["Cents"] == 950)
    b = df["ProductCD"] == "W"
    return (a & b).astype(np.uint8)


def parse_email_suffix(email):
    if not isinstance(email, str) or not email.strip():
        return (np.nan, np.nan, np.nan)

    parts = email.split(".")

    middle = parts[-2] if len(parts) > 2 else parts[-1]

    last = parts[-1]

    return len(parts), middle, last


def str_contains(df: pd.DataFrame, col: str, val: str) -> pd.Series:
    return df[col].str.contains(val, case=False).fillna(False)


# ----------------------------------------------------------------------
# Stage 1: deterministic, row-level features (single-transaction safe)
# ----------------------------------------------------------------------

def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Everything that does NOT require statistics learned from the training
    set. Safe to call on a single incoming transaction (a 1-row dataframe)
    or a full batch — the result is identical either way, row by row.
    """
    df = df.copy()

    # --- Time-based features ---
    seconds = df[COL_DT]
    df["Transaction_Day"] = (seconds // 86400).astype(int)
    df["Transaction_Hour"] = ((seconds % 86400) // 3600).astype(int)
    df["Transaction_Minute"] = ((seconds % 3600) // 60).astype(int)
    df["Transaction_week"] = (df["Transaction_Day"] // 7).astype(int)
    df["Transaction_Time_Of_Day"] = seconds % 86400

    df["Hour_sin"] = np.sin(2 * np.pi * df["Transaction_Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Transaction_Hour"] / 24)

    # --- Amount-based features ---
    df["log_TransactionAmt"] = np.log1p(df[COL_AMOUNT])
    df["AmountBucket"] = pd.cut(
        df[COL_AMOUNT],
        bins=[0, 10, 50, 100, 500, 1000, float("inf")],
        labels=False,
    )

    # --- Missingness features ---
    df["Missing_Count"] = df.isna().sum(axis=1)
    df["Address_Missing"] = (df["addr1"].isna() | df["addr2"].isna()).astype(int)

    # --- Email match ---
    df["email_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype(int)

    # --- Cents-based features ---
    df["Cents"] = feature_cents(df)
    df["Cents_ProductCD_W"] = feature_ProductCD_W_cents(df)

    # --- Interaction (uuid) columns — must exist before group-stat lookups ---
    for inter in INTERACTIONS:
        cols = inter.split("__")
        col_name = RENAME.get(inter, inter)
        df[col_name] = df[cols].apply(lambda row: "_".join(row.values.astype(str)), axis=1)

    # --- Missing-value flags for selected columns ---
    for col in NA_FLAG_COLS:
        df[col + "_na"] = df[col].isna().astype(np.uint8)

    # --- Email domain parsing ---
    for col in ["P_emaildomain", "R_emaildomain"]:
        parsed = df[col].apply(parse_email_suffix)
        df[col + "_parts"] = parsed.map(lambda x: x[0])
        df[col + "_sfx2"] = parsed.map(lambda x: x[1])
        df[col + "_sfx1"] = parsed.map(lambda x: x[2])
        df[col + "_bin"] = df[col].map(EMAIL_MAP)

    # --- Browser / OS extraction from id_31 / id_30 ---
    df["Browser"] = -1
    search_browser = partial(str_contains, df, "id_31")
    for i, browser in enumerate(BROWSERS):
        df.loc[search_browser(browser), ["Browser"]] = i
    df.loc[(df["id_31"] == "ie"), ["Browser"]] = 2

    df["OS"] = -1
    search_os = partial(str_contains, df, "id_30")
    for i, os_name in enumerate(OS_NAMES):
        df.loc[search_os(os_name), ["OS"]] = i

    # --- Binary M-columns: T/F -> 1/0 ---
    for col in M_BINARY_COLS:
        df[col] = df[col].map({"T": 1, "F": 0})

    return df


# ----------------------------------------------------------------------
# Stage 2: training-learned statistics — fit ONCE offline, apply at inference
# ----------------------------------------------------------------------

def fit_training_stats(train_df: pd.DataFrame) -> dict:
    """
    Run ONCE, offline, on the raw training dataframe. Returns a dict of
    lookup tables to be pickled and shipped alongside the trained model.

    Never call this at inference time / inside the deployed app — it must
    only ever see the training distribution.
    """
    base = build_base_features(train_df)

    card1_stats = base.groupby("card1")[COL_AMOUNT].agg(["mean", "std"])

    group_agg_stats = {}
    for group, target, agg in GROUP_AGGS:
        col = f"{target}_{agg}_by_{'_'.join(group)}"
        group_agg_stats[col] = base.groupby(group)[target].agg(agg)

    return {
        "card1_stats": card1_stats,
        "group_agg_stats": group_agg_stats,
    }


def apply_training_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Apply pre-fitted training statistics to a dataframe that has already
    been through `build_base_features`. Only ever does `.map()` /
    MultiIndex lookups against the saved tables — never a live `.groupby()`
    — so this is correct whether `df` has one row or a million.

    Keys not seen during training map to NaN, which CatBoost / LightGBM /
    XGBoost all handle natively as a missing value; that's the right
    behavior here rather than silently substituting a training-time
    average.
    """
    df = df.copy()

    card1_stats = stats["card1_stats"]
    df["TransactionAmt_card1_z"] = (
        (df[COL_AMOUNT] - df["card1"].map(card1_stats["mean"]))
        / (df["card1"].map(card1_stats["std"]) + 1e-5)
    )

    for group, target, agg in GROUP_AGGS:
        col = f"{target}_{agg}_by_{'_'.join(group)}"
        lookup = stats["group_agg_stats"][col]
        if len(group) == 1:
            df[col] = df[group[0]].map(lookup)
        else:
            key = pd.MultiIndex.from_frame(df[group])
            df[col] = key.map(lookup)

    return df


def engineer_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    The single function `inference.py` should call per request:
    deterministic base features, then the saved training-time statistics
    applied via lookup. `stats` must come from `fit_training_stats` (fit
    once, offline) — loaded from disk at app startup, not recomputed here.
    """
    df = build_base_features(df)
    df = apply_training_stats(df, stats)
    return df


# ----------------------------------------------------------------------
# Column alignment against the saved 476-column model schema
# ----------------------------------------------------------------------

def align_columns(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Reindex df to the exact column set AND order the model was trained on
    (e.g. the saved `catboost_feature_columns.pkl`). Raises loudly if a
    required engineered column is missing, rather than letting the model
    silently receive a misaligned row.
    """
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"engineer_features output is missing expected columns: {missing}")
    return df[feature_columns]


# ----------------------------------------------------------------------
# Thin pickle helpers for the stats dict / feature-columns list
# ----------------------------------------------------------------------

def save_stats(stats: dict, path: str) -> None:
    import pickle
    with open(path, "wb") as f:
        pickle.dump(stats, f)


def load_stats(path: str) -> dict:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def load_feature_columns(path: str) -> list:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------
# OFFLINE fitting workflow only — NOT used by the deployed app.
# Run this manually (`python feature_engineering.py`) whenever the model
# is retrained, to regenerate the saved training_stats.pkl. inference.py
# should only ever call `load_stats(...)` + `engineer_features(...)` +
# `align_columns(...)`, never anything below this line.
# ----------------------------------------------------------------------


def _offline_fit_and_save(
    train_transaction_path=DATA_DIR / "train_transaction.txt",
    train_identity_path=DATA_DIR / "train_identity.txt",
    stats_out_path=DATA_DIR / "training_stats.pkl",
):
    train_transaction = pd.read_csv(train_transaction_path)
    train_identity = pd.read_csv(train_identity_path)
    train_df = train_transaction.merge(train_identity, on="TransactionID", how="left")

    stats = fit_training_stats(train_df)
    save_stats(stats, stats_out_path)
    print(f"Saved training-learned stats to {stats_out_path}")


if __name__ == "__main__":
    _offline_fit_and_save()
