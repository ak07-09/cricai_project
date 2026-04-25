"""
train_pipeline.py  (FIXED)
===========================
Orchestrates the full fixed training pipeline:

    DataBuilder  →  FeatureEngineering  →  DataIngestion
    →  DataTransformation  →  ModelTrainer

Run this file directly:
    python src/pipeline/train_pipeline.py

All fixes are documented in the respective component files.
Summary of pipeline-level changes:

  • DataBuilder now emits match_id + season for match-wise split
  • DataBuilder defaults to per-ball snapshots (snapshot_every_n_balls=1)
  • DataBuilder excludes overs 18-20 (max_legal_balls=108) [FIX-DB-1]
  • DataIngestion uses temporal match-wise split [FIX-DI-1, FIX-DI-2]
  • FeatureEngineering drops redundant/quasi-leaky features [FIX-FE-1]
  • ModelTrainer uses Platt calibration + Brier-score CV [FIX-MT-2, FIX-MT-6]
"""

import os
import sys
import time
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.components.data_builder       import DataBuilder,       DataBuilderConfig
from src.components.feature_engineering import FeatureEngineering, FeatureEngineeringConfig
from src.components.data_ingestion     import DataIngestion,      DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer      import ModelTrainer,       ModelTrainerConfig
from src.exception import CustomException
from src.logger    import get_logger

logger = get_logger(__name__)


def run_training_pipeline(
    max_matches: int = None,
    snapshot_every_n_balls: int = 1,     # FIX-DB-4: default to per-ball
    max_legal_balls: int = 108,          # FIX-DB-1: exclude overs 18-20
):
    """
    Parameters
    ----------
    max_matches : int or None
        Cap on JSON files parsed.  None = all.  Use 200 for smoke tests.
    snapshot_every_n_balls : int
        1 = every legal delivery (default, recommended).
        6 = one row per over (faster, less data).
    max_legal_balls : int
        Drop snapshots beyond this ball count.
        108 = 18 overs (recommended).  120 = include all overs (not recommended).
    """
    t0 = time.time()
    print("\n" + "🏏" * 30)
    print("   CRICKET WIN PREDICTOR — FIXED TRAINING PIPELINE")
    print("🏏" * 30)
    print(f"\n   snapshot_every_n_balls : {snapshot_every_n_balls}")
    print(f"   max_legal_balls        : {max_legal_balls} (over {max_legal_balls//6})")

    try:
        # ── Step 1: Build ball-by-ball dataset ───────────────────────────────
        builder_cfg = DataBuilderConfig(
            max_matches=max_matches,
            snapshot_every_n_balls=snapshot_every_n_balls,
            max_legal_balls=max_legal_balls,     # FIX-DB-1
        )
        match_states_path = DataBuilder(builder_cfg).build()

        # ── Step 2: Feature engineering ──────────────────────────────────────
        fe_cfg = FeatureEngineeringConfig(input_path=match_states_path)
        features_path = FeatureEngineering(fe_cfg).build_features()

        # ── Step 3: Data ingestion — match-wise temporal split ────────────────
        ing_cfg = DataIngestionConfig(features_path=features_path)
        train_path, test_path = DataIngestion(ing_cfg).initiate_data_ingestion()

        # ── Step 4: Data transformation ───────────────────────────────────────
        train_arr, test_arr, preprocessor_path = (
            DataTransformation().initiate_data_transformation(train_path, test_path)
        )

        # ── Step 5: Model training + calibration ──────────────────────────────
        feature_names = list(pickle.load(open("artifacts/feature_columns.pkl", "rb")))
        test_auc = ModelTrainer().initiate_model_trainer(
            train_arr, test_arr, feature_names
        )

        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"✅ PIPELINE COMPLETE  —  AUC={test_auc:.4f}  —  {elapsed:.1f}s")
        print(f"{'='*60}\n")
        return test_auc

    except CustomException as ce:
        logger.error(str(ce))
        raise
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the cricket win predictor (fixed)")
    parser.add_argument("--max_matches",    type=int, default=None)
    parser.add_argument("--snapshot_every", type=int, default=1,
                        help="1=every ball (default), 6=per over")
    parser.add_argument("--max_overs",      type=int, default=18,
                        help="Exclude snapshots beyond this over (default 18)")
    args = parser.parse_args()
    run_training_pipeline(
        max_matches=args.max_matches,
        snapshot_every_n_balls=args.snapshot_every,
        max_legal_balls=args.max_overs * 6,
    )
