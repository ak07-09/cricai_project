"""
data_ingestion.py  (FIXED)
==========================
Reads features_engineered.csv, applies anti-leakage guards, and produces
proper train / test splits.

═══════════════════════════════════════════════════════════════════
FIX LOG
═══════════════════════════════════════════════════════════════════

FIX-DI-1  Match-wise split replaces random row-level split (CRITICAL)
   The original used sklearn's train_test_split on individual rows.
   Because multiple rows come from the same match (one per delivery),
   a random split can put delivery 1 of match X in train and delivery
   7 of the same match in test.  Both rows share:
     • the same label (batting_team_won)
     • the same target, batting_team, bowling_team, venue
     • highly correlated run/wicket state
   This is cross-contamination: the model "sees the answer" at test time
   through shared context from training rows.  AUC inflates toward 1.0.

   → Fix: group all deliveries from the same match together.
     Use season as the primary temporal sort key (time-based split).
     Within the last 20% of seasons, all matches go to test.
     This simulates real deployment: "trained on old seasons, predicts new."

FIX-DI-2  Season-based temporal ordering (NEW)
   Random split breaks temporal integrity: the model trains on 2023 matches
   and tests on 2015 matches, making evaluation optimistic (older matches
   may be easier).  Sorting by season and taking the most recent 20% as
   test ensures we measure future-generalisation performance.

FIX-DI-3  match_id and season dropped from feature CSV before saving (NEW)
   These columns are used for splitting but must not reach the model.
   They are removed here after the split is performed.
"""

import os
import sys
import hashlib
import pandas as pd
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataIngestionConfig:
    features_path: str = "data/processed/features_engineered.csv"
    train_data_path: str = "artifacts/train.csv"
    test_data_path: str = "artifacts/test.csv"
    raw_data_path: str = "artifacts/data.csv"
    # Fraction of most-recent seasons to hold out as test
    test_season_fraction: float = 0.20


# Columns that are used for splitting but must not enter the model
_SPLIT_KEY_COLS = ["season", "match_id"]


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = None):
        self.config = config or DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Entered data ingestion (FIXED)")
        print("\n" + "=" * 60)
        print("📥 DATA INGESTION (FIXED)")
        print("=" * 60)

        try:
            # ── 1. Load ───────────────────────────────────────────────────────
            print(f"\n   Loading {self.config.features_path} ...")
            df = pd.read_csv(self.config.features_path)
            print(f"   Shape: {df.shape}")

            target_col = "result"
            if target_col not in df.columns:
                raise ValueError(
                    f"Target column '{target_col}' not found. "
                    f"Columns present: {list(df.columns)}"
                )

            # ── 2. Anti-leakage guard ─────────────────────────────────────────
            leakage_columns = [
                "winner", "winning_team", "losing_team",
                "win_margin", "result_margin",
                "final_score", "match_result", "outcome",
            ]
            df = df.drop(columns=[c for c in leakage_columns if c in df.columns])

            # ── 3. Clean label ────────────────────────────────────────────────
            df = df.dropna(subset=[target_col])
            df[target_col] = df[target_col].astype(int)

            unique_labels = sorted(df[target_col].unique())
            if not set(unique_labels).issubset({0, 1}):
                raise ValueError(f"Unexpected label values: {unique_labels}")

            print(f"\n   Label distribution:")
            vc = df[target_col].value_counts().sort_index()
            for v, c in vc.items():
                print(f"      Class {v}: {c:,}  ({c/len(df)*100:.1f}%)")

            # ── 4. Save full raw artifact (before split) ──────────────────────
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False, header=True)

            # ── 5. FIX-DI-1 + FIX-DI-2: Match-wise temporal split ────────────
            print(f"\n   Performing match-wise temporal split [FIX-DI-1, FIX-DI-2] ...")

            if "season" in df.columns and "match_id" in df.columns:
                train_df, test_df = self._temporal_match_split(df, target_col)
            elif "match_id" in df.columns:
                # Fallback: match-wise hash split (no season info)
                train_df, test_df = self._hash_match_split(df, target_col)
            else:
                # Last resort: stratified row split (original behaviour — warns)
                logger.warning(
                    "match_id not in CSV — falling back to stratified row split. "
                    "Re-run DataBuilder to get match_id."
                )
                from sklearn.model_selection import train_test_split
                train_df, test_df = train_test_split(
                    df, test_size=0.20, random_state=42, stratify=df[target_col]
                )

            # ── 6. FIX-DI-3: Drop split keys before saving ───────────────────
            train_df = train_df.drop(columns=[c for c in _SPLIT_KEY_COLS if c in train_df.columns])
            test_df  = test_df.drop(columns=[c for c in _SPLIT_KEY_COLS if c in test_df.columns])

            train_df.to_csv(self.config.train_data_path, index=False, header=True)
            test_df.to_csv(self.config.test_data_path,   index=False, header=True)

            print(f"   ✅ Train : {len(train_df):,} rows")
            print(f"   ✅ Test  : {len(test_df):,} rows")
            print(f"   Train label balance: {train_df[target_col].mean()*100:.1f}% class-1")
            print(f"   Test  label balance: {test_df[target_col].mean()*100:.1f}% class-1")

            print("\n" + "=" * 60)
            print("✅ DATA INGESTION COMPLETE")
            print("=" * 60)

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

    # ── split strategies ──────────────────────────────────────────────────────

    def _temporal_match_split(self, df: pd.DataFrame, target_col: str):
        """
        Sort seasons chronologically, assign the most recent
        `test_season_fraction` of seasons entirely to test,
        everything else to train.  All rows from a match stay together.
        """
        seasons = sorted(df["season"].unique())
        n_test_seasons = max(1, int(len(seasons) * self.config.test_season_fraction))
        test_seasons = set(seasons[-n_test_seasons:])
        train_seasons = set(seasons[:-n_test_seasons])

        train_df = df[df["season"].isin(train_seasons)].copy()
        test_df  = df[df["season"].isin(test_seasons)].copy()

        print(f"   Seasons in train: {sorted(train_seasons)}")
        print(f"   Seasons in test : {sorted(test_seasons)}")

        return train_df, test_df

    def _hash_match_split(self, df: pd.DataFrame, target_col: str):
        """
        Deterministic hash-based match split when season is unavailable.
        80% of matches → train, 20% → test.  All deliveries from a match
        land in the same bucket.
        """
        def _hash_frac(mid: str) -> float:
            return int(hashlib.md5(mid.encode()).hexdigest(), 16) / (2**128)

        match_ids = df["match_id"].unique()
        test_ids  = {m for m in match_ids if _hash_frac(m) < self.config.test_season_fraction}
        train_df  = df[~df["match_id"].isin(test_ids)].copy()
        test_df   = df[ df["match_id"].isin(test_ids)].copy()
        return train_df, test_df
