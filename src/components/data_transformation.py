"""
data_transformation.py  (v3 — OVERCONFIDENCE FIX)
==================================================

FIX-DT-1  Feature list updated to match v3 FeatureEngineering output.
   REMOVED (were causing AUC=1.0):
     - required_run_rate  (label-leaky)
     - rr_ratio           (redundant with rr_delta)
     - chase_feasibility  (pre-computed probability proxy)
     - rrr_difficulty     (linear clone of RRR)
     - pressure_index     (correlated with multiple rate signals)
     - score_vs_par       (correlated with rr_delta when target is known)
     - balls_bowled, overs_completed, par_score  (redundant derivations)

   KEPT (12 numerical + 4 categorical = 16 total features):
     runs, wickets, balls_left, target,
     current_run_rate, rr_delta,
     balls_remaining_frac, wickets_remaining_frac, resource_index,
     match_progress, wickets_in_hand, runs_per_wicket,
     batting_team, bowling_team, venue, match_type
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object

logger = get_logger(__name__)


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = "artifacts/preprocessor.pkl"
    feature_names_path: str = "artifacts/feature_columns.pkl"


# ── v3 feature list (12 numerical — reduced from 18 in v2) ───────────────────
NUMERICAL_FEATURES = [
    # raw state
    "runs",
    "wickets",
    "balls_left",
    "target",
    # pace
    "current_run_rate",
    # rate gap (ONE signal, not five)
    "rr_delta",
    # resources (two components + combined)
    "balls_remaining_frac",
    "wickets_remaining_frac",
    "resource_index",
    # temporal + stability
    "match_progress",
    "wickets_in_hand",
    "runs_per_wicket",
]

CATEGORICAL_FEATURES = ["batting_team", "bowling_team", "venue", "match_type"]

TARGET_COL = "result"


class DataTransformation:
    def __init__(self, config: DataTransformationConfig = None):
        self.config = config or DataTransformationConfig()

    def get_data_transformer_object(self, train_df: pd.DataFrame):
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float32,
                ),
            ),
        ])

        num_cols = [c for c in NUMERICAL_FEATURES if c in train_df.columns]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in train_df.columns]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return preprocessor, num_cols, cat_cols

    def initiate_data_transformation(self, train_path: str, test_path: str):
        print("\n" + "=" * 60)
        print("🔄 DATA TRANSFORMATION (v3 — OVERCONFIDENCE FIX)")
        print("=" * 60)

        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            print(f"\n   Train shape : {train_df.shape}")
            print(f"   Test  shape : {test_df.shape}")

            X_train = train_df.drop(columns=[TARGET_COL])
            y_train = train_df[TARGET_COL].astype(int).values

            X_test = test_df.drop(columns=[TARGET_COL])
            y_test = test_df[TARGET_COL].astype(int).values

            preprocessor, num_cols, cat_cols = self.get_data_transformer_object(X_train)

            print(f"\n   Numerical  features : {len(num_cols)}")
            print(f"   Categorical features: {len(cat_cols)}")

            train_arr = preprocessor.fit_transform(X_train)
            test_arr  = preprocessor.transform(X_test)

            train_arr = np.c_[train_arr, y_train]
            test_arr  = np.c_[test_arr, y_test]

            print(f"\n   Train array : {train_arr.shape}")
            print(f"   Test  array : {test_arr.shape}")

            feature_names = num_cols + cat_cols
            save_object(self.config.feature_names_path, feature_names)
            save_object(self.config.preprocessor_path, preprocessor)

            print(f"\n   Saved preprocessor → {self.config.preprocessor_path}")
            print("\n" + "=" * 60)
            print("✅ DATA TRANSFORMATION COMPLETE")
            print("=" * 60)

            return train_arr, test_arr, self.config.preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)
