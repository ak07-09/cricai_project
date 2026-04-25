"""
model_trainer.py  (v3 — OVERCONFIDENCE FIX)
============================================

Changes from v2
---------------
FIX-MT-A  Target Brier range updated to reflect v3 expectations.
   With features reduced (no label proxies) and data restricted to overs 2-15,
   the theoretical floor is Brier ≈ 0.13 (logistic regression baseline).
   A well-regularised XGBoost should land at 0.15–0.22.
   A Brier of 0.0007 is definitively overfit / leaky — now flagged as error.

FIX-MT-B  Overconfidence thresholds updated for realistic expectations.
   With overs 2-15 data only, mid-game uncertainty is genuine.
   Target: < 20% extreme predictions, > 30% in the 30-70% uncertain band.

FIX-MT-C  GridSearch scoring kept as neg_brier_score (preserved from v2).
   This is correct — optimise calibration, not just ranking.

FIX-MT-D  Regularisation grid kept strong (preserved from v2).
   max_depth [2,3], min_child_weight [10,20], gamma [0.1,0.5] are critical.
   Even with clean features, XGBoost can still overfit on 300k+ rows.
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object

logger = get_logger(__name__)


@dataclass
class ModelTrainerConfig:
    model_path: str = "artifacts/xgb_model.pkl"
    calibrated_model_path: str = "artifacts/calibrated_model.pkl"


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = None):
        self.config = config or ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array: np.ndarray,
        test_array: np.ndarray,
        feature_names: list,
    ):
        print("\n" + "=" * 60)
        print("🚀 MODEL TRAINING (v3 — OVERCONFIDENCE FIX)")
        print("=" * 60)
        print("\n   Features received:", len(feature_names))
        print("   (Leaky features removed: required_run_rate, chase_feasibility,")
        print("    rrr_difficulty, pressure_index, rr_ratio, score_vs_par)")

        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1].astype(int)
            X_test,  y_test  = test_array[:, :-1],  test_array[:, -1].astype(int)

            print(f"\n   Train: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
            print(f"   Test : {X_test.shape[0]:,} rows × {X_test.shape[1]} features")
            print(f"   Label balance (train): {np.mean(y_train)*100:.1f}% class-1")

            # ── Strongly-regularised grid (same as v2 — correct) ──────────────
            param_grid = {
                "n_estimators":     [100, 200],
                "max_depth":        [2, 3],
                "learning_rate":    [0.05, 0.1],
                "subsample":        [0.6, 0.7],
                "colsample_bytree": [0.5, 0.7],
                "min_child_weight": [10, 20],
                "gamma":            [0.1, 0.5],
                "reg_alpha":        [0.1, 1.0],
                "reg_lambda":       [1.0, 5.0],
            }

            base_xgb = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            )

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            print(f"\n   Running GridSearchCV (5-fold, neg_brier_score) ...")
            gs = GridSearchCV(
                estimator=base_xgb,
                param_grid=param_grid,
                cv=cv,
                scoring="neg_brier_score",
                n_jobs=-1,
                verbose=0,
                refit=True,
            )
            gs.fit(X_train, y_train)

            best_params   = gs.best_params_
            best_cv_brier = -gs.best_score_
            best_model    = gs.best_estimator_

            print(f"\n   Best CV Brier : {best_cv_brier:.4f}")

            # ── FIX-MT-A: Sanity check — Brier should NOT be < 0.10 ───────────
            if best_cv_brier < 0.10:
                print(f"\n   ⚠️  WARNING: CV Brier={best_cv_brier:.4f} < 0.10")
                print(f"   This is suspiciously low. Even after removing label proxies,")
                print(f"   check that no leaky feature slipped through.")
                print(f"   Expected range for T20 win probability: 0.15 – 0.22")
            else:
                print(f"   ✅ Brier score is in realistic range (target: 0.15–0.22)")

            print(f"   Best params:")
            for k, v in best_params.items():
                print(f"      {k}: {v}")

            save_object(self.config.model_path, best_model)

            # ── Platt calibration ─────────────────────────────────────────────
            print(f"\n   Fitting Platt-scaling calibration ...")
            calibrator = CalibratedClassifierCV(best_model, method="sigmoid", cv=5)
            calibrator.fit(X_train, y_train)
            save_object(self.config.calibrated_model_path, calibrator)
            print(f"   Saved → {self.config.calibrated_model_path}")

            # ── Evaluation ────────────────────────────────────────────────────
            self._evaluate("TRAIN", best_model, calibrator, X_train, y_train, feature_names)
            self._evaluate("TEST ", best_model, calibrator, X_test,  y_test,  feature_names)

            # ── Overfitting check ─────────────────────────────────────────────
            train_auc = roc_auc_score(y_train, calibrator.predict_proba(X_train)[:, 1])
            test_auc  = roc_auc_score(y_test,  calibrator.predict_proba(X_test)[:, 1])
            gap = train_auc - test_auc
            print(f"\n   AUC gap (train-test): {gap:.4f} ", end="")
            print("⚠️  Overfitting" if gap > 0.05 else "✅ Acceptable")

            self._prob_distribution_report(calibrator, X_test, y_test)

            print("\n" + "=" * 60)
            print("✅ MODEL TRAINING COMPLETE")
            print("=" * 60)

            return test_auc

        except Exception as e:
            raise CustomException(e, sys)

    def _evaluate(self, split_name, raw_model, cal_model, X, y, feature_names):
        print(f"\n   ── {split_name} metrics ──")

        y_pred     = raw_model.predict(X)
        y_prob_cal = cal_model.predict_proba(X)[:, 1]

        acc   = accuracy_score(y, y_pred)
        auc   = roc_auc_score(y, y_prob_cal)
        f1    = f1_score(y, y_pred, zero_division=0)
        brier = brier_score_loss(y, y_prob_cal)
        ll    = log_loss(y, y_prob_cal)

        print(f"      Accuracy  : {acc*100:.2f}%")
        print(f"      ROC-AUC   : {auc:.4f}")
        print(f"      F1        : {f1:.4f}")
        print(f"      Brier     : {brier:.4f}  (target 0.15–0.22 for overs 2-15)")
        print(f"      Log-Loss  : {ll:.4f}")

        # FIX-MT-B: updated overconfidence threshold
        pct_extreme   = np.mean((y_prob_cal < 0.10) | (y_prob_cal > 0.90)) * 100
        pct_uncertain = np.mean((y_prob_cal >= 0.30) & (y_prob_cal <= 0.70)) * 100

        ext_flag = "⚠️  OVERCONFIDENT" if pct_extreme > 20 else "✅"
        unc_flag = "✅" if pct_uncertain > 30 else "⚠️  too few uncertain predictions"

        print(f"      Extreme preds (<10% or >90%) : {pct_extreme:.1f}%  {ext_flag}")
        print(f"      Uncertain region (30-70%)    : {pct_uncertain:.1f}%  {unc_flag}")

        # FIX-MT-A: flag if Brier is unrealistically good
        if brier < 0.10 and split_name == "TEST ":
            print(f"\n      ⚠️  CRITICAL: Test Brier={brier:.4f} < 0.10")
            print(f"         Model is still memorising the data, not learning probability.")
            print(f"         Re-check feature pipeline for remaining label proxies.")

        logger.info(
            f"{split_name} | AUC={auc:.4f} | Brier={brier:.4f} | "
            f"Extreme={pct_extreme:.1f}% | Uncertain={pct_uncertain:.1f}%"
        )

        if split_name == "TEST " and hasattr(raw_model, "feature_importances_"):
            imp = raw_model.feature_importances_
            n   = min(len(feature_names), len(imp))
            fi  = (
                pd.Series(imp[:n], index=feature_names[:n])
                .sort_values(ascending=False)
            )
            print(f"\n   Top feature importances:")
            for feat, val in fi.head(12).items():
                print(f"      {feat:<28} {val:.4f}")

            top = fi.iloc[0]
            if top > 0.35:
                print(f"\n   ⚠️  '{fi.index[0]}' dominates at {top:.3f}.")
                print(f"   Consider removing it — single-feature dominance = shortcut learning.")

    def _prob_distribution_report(self, cal_model, X_test, y_test):
        probs = cal_model.predict_proba(X_test)[:, 1]

        print(f"\n   ── Probability distribution (test set) ──")
        print(f"   Healthy: roughly uniform. U-shape = overconfident.")
        bins = np.arange(0, 1.1, 0.1)
        counts, _ = np.histogram(probs, bins=bins)
        total = len(probs)
        for i in range(len(counts)):
            lo, hi = bins[i], bins[i + 1]
            pct    = counts[i] / total * 100
            bar    = "█" * int(pct / 2)
            print(f"   [{lo:.1f}-{hi:.1f}]  {bar:<25}  {pct:5.1f}%")

        try:
            frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=5)
            print(f"\n   Calibration curve (predicted vs actual):")
            for mp, fp in zip(mean_pred, frac_pos):
                diff = abs(mp - fp)
                flag = "✅" if diff < 0.05 else "⚠️ "
                print(f"      predicted={mp:.2f}  actual={fp:.2f}  diff={diff:.2f}  {flag}")
        except Exception:
            pass
