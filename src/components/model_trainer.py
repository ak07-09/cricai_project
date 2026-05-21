import os
import sys
import warnings
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
from xgboost import XGBClassifier

# Suppress XGBoost's deprecated-parameter warnings cleanly
warnings.filterwarnings("ignore", message=".*use_label_encoder.*")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

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
        print("🚀 MODEL TRAINING (FAST FIX — NO GRIDSEARCH)")
        print("=" * 60)

        try:
            # ── 1. Split ──────────────────────────────────────────────────────
            X_train, y_train = train_array[:, :-1], train_array[:, -1].astype(int)
            X_test,  y_test  = test_array[:, :-1],  test_array[:, -1].astype(int)

            print(f"\n   Train: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
            print(f"   Test : {X_test.shape[0]:,} rows × {X_test.shape[1]} features")
            print(f"   Label balance (train): {y_train.mean()*100:.1f}% class-1")

            best_model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_estimators=150,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=10,
                gamma=0.3,
                reg_alpha=0.5,
                reg_lambda=3.0,
                n_jobs=-1,
                tree_method="hist",
            )

            print("\n   Training model (FAST MODE — no GridSearch)...")
            best_model.fit(X_train, y_train)

            # ── 3. Raw Brier (before calibration) ────────────────────────────
            raw_train_brier = brier_score_loss(
                y_train, best_model.predict_proba(X_train)[:, 1]
            )
            raw_test_brier = brier_score_loss(
                y_test, best_model.predict_proba(X_test)[:, 1]
            )
            print(f"\n   Raw (uncalibrated) Brier — Train: {raw_train_brier:.4f} | "
                  f"Test: {raw_test_brier:.4f}")
            print(f"   Target Brier range: 0.15 – 0.22 (in-play T20)")
            if raw_test_brier < 0.05:
                print(f"     Brier too low — data-level leakage likely still present")
                

            #  4. Save raw model
            save_object(self.config.model_path, best_model)

           
            print("\n   Applying Platt calibration (sigmoid, cv=5)...")
            calibrator = CalibratedClassifierCV(
                best_model,
                method="sigmoid",   # Platt — stable with small calibration sets
                cv=5,               # changed from cv=3 for more stable fit
            )
            calibrator.fit(X_train, y_train)
            save_object(self.config.calibrated_model_path, calibrator)
            print(f"   Calibrated model saved → {self.config.calibrated_model_path}")

           
            self._evaluate("TRAIN", best_model, calibrator, X_train, y_train, feature_names)
            self._evaluate("TEST ", best_model, calibrator, X_test,  y_test,  feature_names)

            
            train_auc = roc_auc_score(y_train, calibrator.predict_proba(X_train)[:, 1])
            test_auc  = roc_auc_score(y_test,  calibrator.predict_proba(X_test)[:, 1])
            gap = train_auc - test_auc
            print(f"\n   AUC gap (train - test): {gap:.4f}", end=" ")
            if gap > 0.05:
                print("  Potential overfitting")
            elif gap < -0.01:
                print("  Test AUC > Train AUC — row-level split leakage suspected")
            else:
                print(" Acceptable")

            
            self._prob_histogram(calibrator, X_test, y_test)

            print("\n" + "=" * 60)
            print(" MODEL TRAINING COMPLETE")
            print("=" * 60)

            return test_auc

        except Exception as e:
            raise CustomException(e, sys)

    #  helpers

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
        print(f"      Brier     : {brier:.4f}  (target 0.15–0.22)")
        print(f"      Log-Loss  : {ll:.4f}")

        # Overconfidence check
        pct_extreme   = np.mean((y_prob_cal < 0.10) | (y_prob_cal > 0.90)) * 100
        pct_uncertain = np.mean((y_prob_cal > 0.30) & (y_prob_cal < 0.70)) * 100
        flag_e = "⚠️  OVERCONFIDENT" if pct_extreme > 15 else "✅"
        flag_u = "✅" if pct_uncertain > 20 else "⚠️  too few uncertain predictions"
        print(f"      Extreme preds (<10%/>90%) : {pct_extreme:.1f}%  {flag_e}")
        print(f"      Uncertain region (30-70%) : {pct_uncertain:.1f}%  {flag_u}")

        # Feature importance (test set only)
        if split_name == "TEST " and hasattr(raw_model, "feature_importances_"):
            imp = raw_model.feature_importances_
            n   = min(len(feature_names), len(imp))
            fi  = (
                pd.Series(imp[:n], index=feature_names[:n])
                .sort_values(ascending=False)
            )
            print(f"\n   Top-10 feature importances:")
            for feat, val in fi.head(10).items():
                bar = "█" * int(val * 100)
                print(f"      {feat:<28} {val:.4f}  {bar}")
            if fi.iloc[0] > 0.25:
                print(f"\n   ⚠️  '{fi.index[0]}' importance={fi.iloc[0]:.3f} > 0.25 — "
                      f"feature dominance or data leakage")

        logger.info(
            f"{split_name} | AUC={auc:.4f} | Brier={brier:.4f} | "
            f"Extreme={pct_extreme:.1f}% | Uncertain={pct_uncertain:.1f}%"
        )

    def _prob_histogram(self, cal_model, X_test, y_test):
        """
        Print a text histogram of predicted probabilities.
        A healthy model → roughly uniform distribution.
        An overconfident model → U-shaped (spikes at 0 and 1).
        """
        probs = cal_model.predict_proba(X_test)[:, 1]
        total = len(probs)

        print(f"\n   ── Probability distribution (test set) ──")
        print(f"   (Healthy = roughly uniform | Overconfident = U-shaped spikes)")
        bins = np.arange(0, 1.1, 0.1)
        counts, _ = np.histogram(probs, bins=bins)
        for i in range(len(counts)):
            lo, hi  = bins[i], bins[i + 1]
            pct     = counts[i] / total * 100
            bar     = "█" * int(pct / 2)
            spike   = " ← SPIKE ⚠️" if pct > 25 and (lo < 0.1 or hi > 0.9) else ""
            print(f"   [{lo:.1f}-{hi:.1f}]  {bar:<25}  {pct:5.1f}%{spike}")

        # Calibration curve
        try:
            frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=5)
            print(f"\n   Calibration curve (mean predicted vs actual win rate):")
            for mp, fp in zip(mean_pred, frac_pos):
                diff = abs(mp - fp)
                flag = "✅" if diff < 0.05 else "⚠️ "
                print(f"      predicted={mp:.2f}  actual={fp:.2f}  diff={diff:.2f}  {flag}")
        except Exception:
            pass