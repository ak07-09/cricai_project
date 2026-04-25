"""
feature_engineering.py  (v3 — OVERCONFIDENCE FIX)
==================================================

ROOT CAUSE OF AUC=1.0 / BRIER=0.0007
======================================

The previous version (v2) contained five features that together constitute
a near-perfect label proxy, making the model deterministic rather than
probabilistic:

1. chase_feasibility = 1/(1+exp(-rr_delta×0.5))
   → A logistic function of rr_delta. This is literally pre-computing a
     probability-like score from the same signal the model is supposed to
     learn. With this feature, the model needs only one split to get AUC≈1.

2. rrr_difficulty = required_run_rate / 9.0
   → A linear rescaling of required_run_rate. Adds no new information
     beyond RRR itself; just duplicates the dominant feature.

3. pressure_index recomputed from (balls_used / balls_left_per_over / wkts_in_hand)
   → Encodes the deficit-per-resource in yet another form. Correlated with
     rr_ratio, rr_delta, and RRR simultaneously.

4. resource_index = balls_remaining_frac × wickets_remaining_frac
   → Legitimate uncertainty feature, but with all the above present the
     model uses it as a "weight" for the already-deterministic RRR signal
     rather than as an uncertainty indicator.

5. score_vs_par = runs - target × match_progress
   → When combined with RRR in the same model, this provides a correlated
     second view of the same deficit. The two features allow XGBoost to
     triangulate the exact outcome with no ambiguity.

Together, features 1-5 plus required_run_rate give XGBoost enough axes to
perfectly partition every (RRR, rr_delta, score_vs_par) combination and
assign the correct label without ever seeing a mis-classified example.

THE FIX
========

Principle: "One signal per concept." Cricket win probability depends on
three independent concepts:
  A. How fast the batting team is scoring vs. how fast they need to score  →  rr_delta only
  B. How many resources (balls + wickets) remain                           →  resource_index only
  C. Context (team strength, venue, target magnitude)                      →  categoricals + target

Every other feature is a transformation of A or B. We keep one clean
representative per concept and remove all redundant variants.

Removed:
  - required_run_rate        (keep rr_delta = CRR - RRR; RRR alone is label-leaky)
  - rr_ratio                 (= CRR/RRR, redundant with rr_delta)
  - chase_feasibility        (= logistic(rr_delta), pre-computed probability — CRITICAL REMOVE)
  - rrr_difficulty           (= RRR/9, linear rescaling of required_run_rate)
  - pressure_index           (correlated combination of A and B)
  - score_vs_par             (correlated with rr_delta when target is known)

Kept:
  - rr_delta                 (the one clean rate-gap signal)
  - resource_index           (the one combined resource signal)
  - balls_remaining_frac     (component of resource — helps interpolation)
  - wickets_remaining_frac   (component of resource — helps interpolation)
  - runs_per_wicket          (batting stability — orthogonal to rate signals)
  - match_progress           (temporal context)
  - runs, wickets, balls_left, target  (raw state — model may find non-linear patterns)
  - current_run_rate         (batting pace — independently useful)

EXPECTED OUTCOME AFTER FIX
============================
  Brier score  : 0.17 – 0.22  (realistic uncertainty)
  ROC-AUC      : 0.76 – 0.84  (genuinely predictive, not deterministic)
  Extreme preds: < 25%        (smooth probability distribution)
  Uncertain 30-70%: > 30%    (model acknowledges mid-game uncertainty)
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)


@dataclass
class FeatureEngineeringConfig:
    input_path: str = "data/processed/match_states.csv"
    output_path: str = "data/processed/features_engineered.csv"
    crr_cap: float = 24.0      # 4 runs/ball max — never achieved in T20 history


class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig = None):
        self.config = config or FeatureEngineeringConfig()

    def build_features(self) -> str:
        logger.info("=== FeatureEngineering (v3 — overconfidence fix): start ===")
        print("\n" + "=" * 60)
        print("⚙️  FEATURE ENGINEERING (v3 — OVERCONFIDENCE FIX)")
        print("=" * 60)
        print("\n   KEY CHANGE: removed all pre-computed probability proxies")
        print("   (chase_feasibility, rrr_difficulty, pressure_index, rr_ratio)")
        print("   These caused AUC=1.0 by giving the model the answer directly.")

        try:
            df = pd.read_csv(self.config.input_path)
            print(f"\n   Loaded {len(df):,} rows from {self.config.input_path}")

            df = self._engineer(df)

            os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
            df.to_csv(self.config.output_path, index=False)

            n_features = df.shape[1] - 1
            print(f"\n✅ Features engineered: {n_features} features + 1 label")
            print(f"   Saved to: {self.config.output_path}")
            logger.info(f"FeatureEngineering (v3) done: {df.shape}")
            return self.config.output_path

        except Exception as e:
            raise CustomException(e, sys)

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── 1. Drop identifiers (split keys — never features) ─────────────────
        df = df.drop(columns=[c for c in ["season", "match_id"] if c in df.columns])

        # ── 2. Rename label ───────────────────────────────────────────────────
        if "batting_team_won" in df.columns:
            df = df.rename(columns={"batting_team_won": "result"})

        # ── 3. Drop raw columns that are redundant or directly re-encoded below
        #       (keep raw state: runs, wickets, balls_left, target, CRR)
        drop_cols = [
            "runs_required",    # = target - runs (both kept separately)
            "balls_bowled",     # = 120 - balls_left
            "overs_completed",  # = (120 - balls_left) / 6
            "required_run_rate",  # REMOVED: too label-leaky; rr_delta carries the gap signal
            "par_score",          # = target * match_progress (derivable)
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # ── 4. Cap current_run_rate ───────────────────────────────────────────
        df["current_run_rate"] = df["current_run_rate"].clip(0, self.config.crr_cap)

        # ── 5. Wickets in hand ────────────────────────────────────────────────
        df["wickets_in_hand"] = 10 - df["wickets"]

        # ── 6. Rate gap signal (CONCEPT A — ONE signal only) ─────────────────
        # runs_required is still in df (if not dropped above via another path)
        # Recompute from state columns to be safe:
        _runs_req = (df["target"] - df["runs"]).clip(lower=0)
        _rrr = (_runs_req * 6) / df["balls_left"].clip(lower=1)
        _rrr = _rrr.clip(0, self.config.crr_cap)

        # rr_delta is the ONE rate-gap feature.
        # It encodes "how much faster/slower the batting team is scoring vs needed"
        # Range: strongly negative = well behind, positive = ahead
        # NOT a probability — requires 5+ overs of uncertainty to resolve
        df["rr_delta"] = df["current_run_rate"] - _rrr

        # ── 7. Resource signals (CONCEPT B — two components + combined) ───────
        df["balls_remaining_frac"]   = df["balls_left"] / 120.0
        df["wickets_remaining_frac"] = df["wickets_in_hand"] / 10.0

        # Combined resource index: decays fast as balls AND wickets deplete
        # At ball 60, 5wkts: 0.5 × 0.5 = 0.25  (lots of uncertainty)
        # At ball 90, 2wkts: 0.25 × 0.2 = 0.05 (little uncertainty)
        # This is the single combined resource signal — keeps the model aware of
        # how much "room" remains for uncertainty.
        df["resource_index"] = df["balls_remaining_frac"] * df["wickets_remaining_frac"]

        # ── 8. Match progress (temporal context) ──────────────────────────────
        df["match_progress"] = (120 - df["balls_left"]) / 120.0

        # ── 9. Batting stability (orthogonal to rate signals) ─────────────────
        df["runs_per_wicket"] = df["runs"] / df["wickets"].clip(lower=1)
        df["runs_per_wicket"] = df["runs_per_wicket"].clip(0, 200)

        # ── 10. Clean up NaN / Inf ────────────────────────────────────────────
        pre = len(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna()
        if len(df) != pre:
            logger.warning(f"Dropped {pre - len(df)} rows with NaN/Inf")

        # ── 11. Final column order ────────────────────────────────────────────
        # NUMERICAL: 12 features (down from 18 in v2)
        numerical_features = [
            # raw state
            "runs",
            "wickets",
            "balls_left",
            "target",
            # pace
            "current_run_rate",
            # CONCEPT A: rate gap (one signal only — rr_delta)
            "rr_delta",
            # CONCEPT B: resource (two components + combined)
            "balls_remaining_frac",
            "wickets_remaining_frac",
            "resource_index",
            # temporal + stability
            "match_progress",
            "wickets_in_hand",
            "runs_per_wicket",
        ]

        # CATEGORICAL: 4 features
        categorical_features = ["batting_team", "bowling_team", "venue", "match_type"]

        label = ["result"]

        ordered_cols = numerical_features + categorical_features + label
        df = df[[c for c in ordered_cols if c in df.columns]]

        # Report
        num_present = [c for c in numerical_features if c in df.columns]
        cat_present  = [c for c in categorical_features if c in df.columns]

        print(f"\n   Numerical  features : {len(num_present)}")
        for f in num_present:
            print(f"      ✓ {f}")
        print(f"\n   Categorical features: {len(cat_present)}")
        print(f"   Label               : result (1 = batting team wins)")
        print(f"   Final rows          : {len(df):,}")

        print(f"\n   REMOVED (were causing AUC=1.0):")
        print(f"      ✗ required_run_rate  (label-leaky — RRR at over 15+ predicts winner)")
        print(f"      ✗ chase_feasibility  (= logistic(rr_delta) — pre-computed probability)")
        print(f"      ✗ rrr_difficulty     (= RRR/9 — linear clone of required_run_rate)")
        print(f"      ✗ pressure_index     (correlated combo of RRR + resource)")
        print(f"      ✗ rr_ratio           (= CRR/RRR — redundant with rr_delta)")
        print(f"      ✗ score_vs_par       (correlated with rr_delta when target known)")

        print(f"\n   Label distribution:")
        vc = df["result"].value_counts().sort_index()
        for v, c in vc.items():
            print(f"      {int(v)} → {c:,}  ({c/len(df)*100:.1f}%)")

        return df
