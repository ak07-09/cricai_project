"""
data_builder.py  (FIXED)
========================
Builds a ball-by-ball match-state dataset from raw Cricsheet JSON files.

═══════════════════════════════════════════════════════════════════
FIX LOG — what this file corrects vs. the submitted version
═══════════════════════════════════════════════════════════════════

FIX-DB-1  Late-game snapshot exclusion (NEW)
   The original emitted snapshots all the way to ball 120 (the last legal
   delivery).  In overs 18-20 the match outcome is almost always settled,
   so rows near the end always carry extreme labels (0 or 1) with no
   uncertainty.  The model learns "late game → certainty" and applies that
   shortcut everywhere.
   → Added  max_legal_balls  config (default 108 = 18 overs).
     Snapshots after ball 108 are silently dropped so the model never
     trains on "certain" end-states.

FIX-DB-2  Season kept as a usable split key (PRESERVED)
   The original version included `season` in the CSV.  This is correct:
   DataIngestion uses it for time-based splitting (see data_ingestion.py).
   We keep it but ensure it is NOT passed to the feature matrix.

FIX-DB-3  match_id added to CSV (NEW)
   A unique match identifier is essential for match-wise grouping during
   the train/test split (FIX-DI-1 in data_ingestion.py).  Without it,
   rows from the same match can end up in both train and test, creating
   label leakage across deliveries.
   → Derived from filename stem (Cricsheet convention: date_teams.json).

FIX-DB-4  snapshot_every_n_balls default changed to 1 (CHANGED)
   Per-over snapshots (every 6 balls) discard granular mid-over state
   variation and reduce training data 6-fold.  Ball-by-ball data gives
   the model richer exposure to every pressure situation.
   → Default is now 1 (every legal delivery).  Set to 6 for fast runs.

FIX-DB-5  min_legal_balls raised to 12 (CHANGED)
   Rows before the end of over 2 contain almost no meaningful state
   (score 0-5, target unchanged).  The original min of 6 let the very
   first over through; raising to 12 (end of over 2) reduces noise.

No features are pre-computed here beyond raw match state — all derived
features live in feature_engineering.py so changes propagate correctly
to inference without touching the builder.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)


@dataclass
class DataBuilderConfig:
    international_dir: str = "data/raw/international"
    ipl_dir: str = "data/raw/ipl"
    output_path: str = "data/processed/match_states.csv"
    match_types: List[str] = field(default_factory=lambda: ["T20"])

    # FIX-DB-4: every legal ball for maximum granularity
    snapshot_every_n_balls: int = 1

    # FIX-DB-5: skip the first 2 overs (too little state info)
    min_legal_balls: int = 12

    # FIX-DB-1: drop snapshots after over 18 (outcome almost certain)
    max_legal_balls: int = 90      # 15 overs × 6 balls (FIX: over 16-20 RRR is near-deterministic)

    max_matches: Optional[int] = None


class DataBuilder:
    """
    Parses raw Cricsheet JSON files and produces a clean ball-by-ball
    match-state CSV ready for feature engineering.
    """

    def __init__(self, config: DataBuilderConfig = None):
        self.config = config or DataBuilderConfig()

    # ── public entry point ────────────────────────────────────────────────────

    def build(self) -> str:
        """Parse all JSON files and write match_states.csv. Returns output path."""
        logger.info("=== DataBuilder (FIXED): starting JSON parse ===")
        print("\n" + "=" * 60)
        print("📂 DATA BUILDER (FIXED) — Ball-by-Ball Parser")
        print("=" * 60)
        print(f"   Snapshot every  : {self.config.snapshot_every_n_balls} legal ball(s)")
        print(f"   Min legal balls : {self.config.min_legal_balls}  (over {self.config.min_legal_balls//6})")
        print(f"   Max legal balls : {self.config.max_legal_balls}  (over {self.config.max_legal_balls//6})  [FIX-DB-1]")

        all_files = self._collect_files()
        print(f"   Total JSON files: {len(all_files)}")
        if self.config.max_matches:
            all_files = all_files[: self.config.max_matches]

        rows, skipped, parsed = [], 0, 0
        for path in all_files:
            try:
                new_rows = self._parse_match(path)
                rows.extend(new_rows)
                parsed += 1
            except Exception as exc:
                logger.debug(f"Skipped {path}: {exc}")
                skipped += 1

        if not rows:
            raise CustomException(
                "No valid rows extracted. Check JSON files contain T20 matches "
                "with two complete innings and a declared winner.",
                sys,
            )

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
        df.to_csv(self.config.output_path, index=False)

        print(f"\n✅ Parsed   : {parsed} matches")
        print(f"   Skipped  : {skipped} matches")
        print(f"   Rows     : {len(df):,}")
        print(f"   Label=1  : {df['batting_team_won'].mean()*100:.1f}% (chaser won)")
        print(f"   Saved to : {self.config.output_path}\n")
        logger.info(f"DataBuilder complete: {len(df):,} rows → {self.config.output_path}")
        return self.config.output_path

    # ── private helpers ───────────────────────────────────────────────────────

    def _collect_files(self) -> List[str]:
        files = []
        for d in [self.config.international_dir, self.config.ipl_dir]:
            if os.path.isdir(d):
                files.extend(sorted(glob.glob(os.path.join(d, "*.json"))))
        return files

    def _parse_match(self, path: str) -> List[dict]:
        with open(path) as f:
            data = json.load(f)

        info = data.get("info", {})
        if info.get("match_type") not in self.config.match_types:
            return []

        outcome = info.get("outcome", {})
        winner = outcome.get("winner")
        if not winner:
            return []

        teams = info.get("teams", [])
        venue = info.get("venue", "Unknown")
        match_type = info.get("match_type", "T20")
        season = str(info.get("season", "Unknown"))

        # FIX-DB-3: derive a stable match_id from the file name
        match_id = os.path.splitext(os.path.basename(path))[0]

        innings_list = data.get("innings", [])
        if len(innings_list) < 2:
            return []

        inn1 = innings_list[0]
        inn2 = innings_list[1]

        inn1_runs, _, _ = self._sum_innings(inn1)
        target = inn1_runs + 1

        batting_team = inn2.get("team", "Unknown")
        bowling_team = inn1.get("team", "Unknown")
        batting_team_won = 1 if winner == batting_team else 0

        rows = self._build_rows(
            inn2, target, batting_team, bowling_team,
            venue, match_type, season, match_id, batting_team_won
        )
        return rows

    def _sum_innings(self, inn: dict):
        runs = wickets = legal_balls = 0
        for ov_data in inn.get("overs", []):
            for ball in ov_data.get("deliveries", []):
                runs += ball["runs"]["total"]
                extras = ball.get("extras", {})
                is_wide = "wides" in extras
                is_nb = "noballs" in extras
                if not (is_wide or is_nb):
                    legal_balls += 1
                if "wickets" in ball:
                    w_list = ball["wickets"]
                    wickets += len(w_list) if isinstance(w_list, list) else 1
        return runs, wickets, legal_balls

    def _build_rows(
        self,
        inn: dict,
        target: int,
        batting_team: str,
        bowling_team: str,
        venue: str,
        match_type: str,
        season: str,
        match_id: str,           # FIX-DB-3
        batting_team_won: int,
    ) -> List[dict]:

        cum_runs = 0
        cum_wickets = 0
        legal_balls = 0
        rows = []

        for ov_data in inn.get("overs", []):
            for ball in ov_data.get("deliveries", []):
                extras = ball.get("extras", {})
                is_wide = "wides" in extras
                is_nb = "noballs" in extras

                cum_runs += ball["runs"]["total"]
                if "wickets" in ball:
                    w_list = ball["wickets"]
                    cum_wickets += len(w_list) if isinstance(w_list, list) else 1
                if not (is_wide or is_nb):
                    legal_balls += 1

                # FIX-DB-1: skip late-game rows (over 18+)
                if legal_balls > self.config.max_legal_balls:
                    continue

                if (
                    legal_balls >= self.config.min_legal_balls
                    and legal_balls % self.config.snapshot_every_n_balls == 0
                ):
                    row = self._make_row(
                        legal_balls, cum_runs, cum_wickets, target,
                        batting_team, bowling_team, venue,
                        match_type, season, match_id, batting_team_won
                    )
                    rows.append(row)

        return rows

    @staticmethod
    def _make_row(
        legal_balls: int,
        runs: int,
        wickets: int,
        target: int,
        batting_team: str,
        bowling_team: str,
        venue: str,
        match_type: str,
        season: str,
        match_id: str,           # FIX-DB-3
        batting_team_won: int,
    ) -> dict:
        balls_left = max(1, 120 - legal_balls)
        overs_done = legal_balls / 6
        runs_req = max(0, target - runs)
        rrr = (runs_req * 6) / balls_left
        crr = (runs * 6) / legal_balls if legal_balls > 0 else 0.0

        return {
            # identifiers — used for splitting, dropped before training
            "season":          season,
            "match_id":        match_id,       # FIX-DB-3
            # live match state
            "batting_team":    batting_team,
            "bowling_team":    bowling_team,
            "venue":           venue,
            "match_type":      match_type,
            "runs":            runs,
            "wickets":         wickets,
            "balls_bowled":    legal_balls,
            "overs_completed": round(overs_done, 4),
            "target":          target,
            "runs_required":   runs_req,
            "balls_left":      balls_left,
            "current_run_rate":  round(crr, 4),
            "required_run_rate": round(rrr, 4),
            # label
            "batting_team_won": batting_team_won,
        }
