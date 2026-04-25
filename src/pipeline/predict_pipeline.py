"""
predict_pipeline.py  (v3 — OVERCONFIDENCE FIX)
===============================================
Feature construction mirrors v3 FeatureEngineering EXACTLY.

REMOVED (were causing AUC=1.0):
  required_run_rate, rr_ratio, chase_feasibility,
  rrr_difficulty, pressure_index, score_vs_par

KEPT (12 numerical + 4 categorical = 16 features):
  runs, wickets, balls_left, target,
  current_run_rate, rr_delta,
  balls_remaining_frac, wickets_remaining_frac, resource_index,
  match_progress, wickets_in_hand, runs_per_wicket,
  batting_team, bowling_team, venue, match_type
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils import load_object
from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

CRR_CAP = 24.0
TOTAL_LEGAL_BALLS = 120


class PredictionPipeline:
    def __init__(self):
        try:
            self.model         = load_object("artifacts/calibrated_model.pkl")
            self.preprocessor  = load_object("artifacts/preprocessor.pkl")
            self.feature_names = load_object("artifacts/feature_columns.pkl")
            logger.info("Artifacts loaded")
        except FileNotFoundError as e:
            raise CustomException(f"Artifact not found ({e}). Run training pipeline first.", sys)
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features_dict: dict) -> dict:
        try:
            runs    = int(features_dict.get("current_score", 0))
            wickets = int(features_dict.get("wickets", 0))
            target  = int(features_dict.get("target", 180))
            overs_raw   = float(features_dict.get("overs_completed", 0))
            legal_balls = self._overs_to_legal_balls(overs_raw)

            if wickets >= 10:
                return {"batting_win": 0.0, "bowling_win": 100.0}
            if runs >= target:
                return {"batting_win": 100.0, "bowling_win": 0.0}

            balls_left      = max(1, TOTAL_LEGAL_BALLS - legal_balls)
            wickets_in_hand = 10 - wickets
            runs_req        = max(0, target - runs)

            crr  = min((runs * 6) / legal_balls if legal_balls > 0 else 0.0, CRR_CAP)
            _rrr = min((runs_req * 6) / balls_left, CRR_CAP)
            rr_delta = crr - _rrr

            balls_remaining_frac   = balls_left / 120.0
            wickets_remaining_frac = wickets_in_hand / 10.0
            resource_index         = balls_remaining_frac * wickets_remaining_frac
            match_progress         = (TOTAL_LEGAL_BALLS - balls_left) / TOTAL_LEGAL_BALLS
            runs_per_wicket        = min(runs / max(wickets, 1), 200.0)

            row = {
                "runs": runs, "wickets": wickets, "balls_left": balls_left, "target": target,
                "current_run_rate": round(crr, 4), "rr_delta": round(rr_delta, 4),
                "balls_remaining_frac": round(balls_remaining_frac, 4),
                "wickets_remaining_frac": round(wickets_remaining_frac, 4),
                "resource_index": round(resource_index, 4),
                "match_progress": round(match_progress, 4),
                "wickets_in_hand": wickets_in_hand,
                "runs_per_wicket": round(runs_per_wicket, 4),
                "batting_team": features_dict.get("batting_team", "Unknown"),
                "bowling_team": features_dict.get("bowling_team", "Unknown"),
                "venue":        features_dict.get("venue", "Unknown"),
                "match_type":   features_dict.get("match_type", "T20"),
            }

            X             = pd.DataFrame([row])
            X_transformed = self.preprocessor.transform(X)
            prob          = float(self.model.predict_proba(X_transformed)[0, 1])
            prob          = max(0.02, min(0.98, prob))

            return {"batting_win": round(prob * 100, 2), "bowling_win": round((1 - prob) * 100, 2)}

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _overs_to_legal_balls(overs: float) -> int:
        full = int(overs)
        extra = min(round((overs - full) * 10), 5)
        return full * 6 + extra
