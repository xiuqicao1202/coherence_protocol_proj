
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
all.py — Integrate 1run/2run/3run signals, evaluate every 10 seconds and output weighted total score.
Output: all.csv, including
timestamp, operator_id, heart_rate_bpm, score1, score2, score3, score_all, message, events, comments

Description:
- score1 (0-100, higher is better): Based on heart rate RMSSD (HRV) + heart rate z-score penalty
- score2 (0-100, higher is better): Chat positivity/consistency (keywords + momentum smoothing)
- score3 (0-100, higher is better): Task health (completion rate↑, error rate↓, cycle time↓ + momentum smoothing)
- score_all: Weighted sum (default 0.34/0.33/0.33)

If the current score_all is lower than the historical (all people, all time, strictly before current) mean - 1σ,
then check if any of the three scores < its "class historical mean - 1σ" (all people, all time, strictly before current, and only compared within the class).
If so, give a reason and a little context in comments.
"""

from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
from datetime import timezone

import sys
from dictionary import DictionaryScorer

class TeeLogger:
    """
    TeeLogger: Print output to both stdout and all.log
    """
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

# Clear log file first, then redirect stdout
with open("all.log", "w", encoding="utf-8"):
    pass  # Only for clearing content
sys.stdout = TeeLogger("all.log")

BASE = Path(".")

def read_csv_safely(path: Path) -> pd.DataFrame:
    """
    Safely read csv file, return empty DataFrame if not exist, compatible with different pandas error arguments.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(path, error_bad_lines=False)

def load_inputs(base: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read three types of input data (heart rate, chat, task), check basic fields and convert timestamps.
    """
    hr = read_csv_safely(base / "1input.csv")
    chat = read_csv_safely(base / "2input.csv")
    tasks = read_csv_safely(base / "3input.csv")
    # Convert timestamp to UTC
    for df in [hr, chat, tasks]:
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    # Check fields, fill with empty table if missing
    if not {"timestamp", "operator_id", "heart_rate_bpm"}.issubset(set(hr.columns)):
        hr = pd.DataFrame(columns=["timestamp", "operator_id", "heart_rate_bpm"])
    if not {"timestamp", "operator_id", "message"}.issubset(set(chat.columns)):
        chat = pd.DataFrame(columns=["timestamp", "operator_id", "message"])
    if not {"timestamp", "operator_id", "event", "task_id"}.issubset(set(tasks.columns)):
        tasks = pd.DataFrame(columns=["timestamp", "operator_id", "event", "task_id"])
    return hr, chat, tasks

def make_time_bins(hr: pd.DataFrame, chat: pd.DataFrame, tasks: pd.DataFrame, bin_seconds: int = 10) -> Tuple[List[pd.Timestamp], List[str]]:
    """
    Calculate the time range of all data, generate a timeline with every bin_seconds, and all operator_id appeared.
    """
    dfs = [df for df in [hr, chat, tasks] if not df.empty]
    if not dfs:
        return [], []
    # Get min/max time of all data
    tmin = min(df["timestamp"].min() for df in dfs if not df["timestamp"].isna().all())
    tmax = max(df["timestamp"].max() for df in dfs if not df["timestamp"].isna().all())
    print('tmin, tmax', tmin, tmax)
    if pd.isna(tmin) or pd.isna(tmax):
        return [], []
    # Align to bin boundary
    tmin = (tmin - pd.Timedelta(seconds=tmin.second % bin_seconds, microseconds=tmin.microsecond)).floor(f"{bin_seconds}s")
    tmax = (tmax + pd.Timedelta(seconds=(bin_seconds - (tmax.second % bin_seconds)) % bin_seconds)).ceil(f"{bin_seconds}s")
    timeline = list(pd.date_range(tmin, tmax, freq=f"{bin_seconds}s", tz=timezone.utc))
    # Collect all operator_id
    ops = sorted(set(hr.get("operator_id", pd.Series(dtype=str)).dropna().unique().tolist()
                     + chat.get("operator_id", pd.Series(dtype=str)).dropna().unique().tolist()
                     + tasks.get("operator_id", pd.Series(dtype=str)).dropna().unique().tolist()))

    print('timeline', timeline)
    print('ops', ops)
    return timeline, ops

def assign_bins(df: pd.DataFrame, bin_seconds: int = 10) -> pd.DataFrame:
    """
    Assign each row to its time bin (floor).
    """
    if df.empty:
        return df
    df = df.copy()
    df["bin_ts"] = df["timestamp"].dt.floor(f"{bin_seconds}s")
    return df

def compute_scores(hr: pd.DataFrame, chat: pd.DataFrame, tasks: pd.DataFrame,
                   timeline: List[pd.Timestamp], ops: List[str],
                   bin_seconds: int = 10, weights: Tuple[float, float, float] = (0.34, 0.33, 0.33)) -> pd.DataFrame:
    """
    Main scoring process: aggregate data, calculate three scores, weighted total, generate comments.
    """
    # 1. Aggregate three types of data by bin
    bhr = assign_bins(hr, bin_seconds)
    bchat = assign_bins(chat, bin_seconds)
    btasks = assign_bins(tasks, bin_seconds)
    # Heart rate aggregation: mean per bin
    hr_agg = bhr.groupby(["bin_ts", "operator_id"], as_index=False).agg(
        heart_rate_bpm=("heart_rate_bpm", "mean")
    )

    # Aggregate chat data by bin_ts and operator_id, concatenate all messages in each bin (" | " separated), only include data at or before bin_ts
    # For example, bin 2025-08-20 10:00:10+00:00 only includes original data with timestamp <= 10:00:10
    # So each bin only reflects content up to that time, not after
    chat_agg = []
    chat_by_op = chat.groupby("operator_id")
    for bin_ts in timeline:
        for op in ops:
            if op in chat_by_op.groups:
                op_chat = chat_by_op.get_group(op)
                start_time = bin_ts - pd.Timedelta(seconds=10)
                end_time = bin_ts
                mask = (op_chat["timestamp"] > start_time) & (op_chat["timestamp"] <= end_time)
                msgs = op_chat[mask]["message"].dropna().tolist()
            else:
                msgs = []
            msg_str = " | ".join(map(str, msgs)) if msgs else np.nan
            chat_agg.append({"bin_ts": bin_ts, "operator_id": op, "message": msg_str})
    chat_agg = pd.DataFrame(chat_agg)
    print('chat_agg', chat_agg)

    # Task aggregation: concatenate event:task_id per bin, same as chat_agg, only include events up to that time
    task_agg = []
    tasks_by_op = btasks.groupby("operator_id")
    for bin_ts in timeline:
        for op in ops:
            if op in tasks_by_op.groups:
                op_tasks = tasks_by_op.get_group(op)
                start_time = bin_ts - pd.Timedelta(seconds=10)
                end_time = bin_ts
                mask = (op_tasks["timestamp"] > start_time) & (op_tasks["timestamp"] <= end_time)
                events = op_tasks[mask]
                event_pairs = [
                    f"{row['event']}:{row['task_id']}"
                    for _, row in events.iterrows()
                    if pd.notna(row.get("event")) and pd.notna(row.get("task_id"))
                ]
            else:
                event_pairs = []
            events_str = " | ".join(event_pairs) if event_pairs else np.nan
            task_agg.append({"bin_ts": bin_ts, "operator_id": op, "events": events_str})
    task_agg = pd.DataFrame(task_agg)

    # Generate full time-operator grid, left join three aggregated data
    grid = pd.MultiIndex.from_product([timeline, ops], names=["bin_ts", "operator_id"]).to_frame(index=False)
    df = grid.merge(hr_agg, on=["bin_ts", "operator_id"], how="left") \
             .merge(chat_agg, on=["bin_ts", "operator_id"], how="left") \
             .merge(task_agg, on=["bin_ts", "operator_id"], how="left")

    window_bins = 6  # Rolling window length (6*10s=1min)
    df = df.sort_values(["operator_id", "bin_ts"]).reset_index(drop=True)

    # ---- score1: HRV + heart rate z-score penalty ----
    # Calculate rolling mean/std/z-score (only use current and previous values, not future)
    # Note: rolling window is right-closed (includes current, not future), center=False
    df["hr_roll_mean"] = df.groupby("operator_id")["heart_rate_bpm"].transform(
        lambda s: s.rolling(window=window_bins, min_periods=3).mean()
    )
    df["hr_roll_std"] = df.groupby("operator_id")["heart_rate_bpm"].transform(
        lambda s: s.rolling(window=window_bins, min_periods=3).std()
    )
    df["hr_z"] = (df["heart_rate_bpm"] - df["hr_roll_mean"]) / df["hr_roll_std"]
    df.loc[df["hr_roll_std"].isna() | (df["hr_roll_std"] == 0), "hr_z"] = np.nan
    df["hr_z_abs"] = df["hr_z"].abs()

    # Calculate rolling RMSSD (also only use current and previous values)
    def rmssd_from_hr(series_bpm: pd.Series) -> float:
        """
        Calculate RMSSD (HRV metric) from a series of heart rates (bpm).
        """
        hr = series_bpm.dropna().astype(float)
        if len(hr) < 3:
            return np.nan
        rr = 60000.0 / hr  # RR interval (ms)
        diff = np.diff(rr)
        if len(diff) == 0:
            return np.nan
        return float(np.sqrt(np.mean(np.square(diff))))
    def rolling_rmssd(s: pd.Series) -> pd.Series:
        out = []
        vals = list(s)
        for i in range(len(vals)):
            start = max(0, i - window_bins + 1)
            out.append(rmssd_from_hr(pd.Series(vals[start:i+1])))
        return pd.Series(out, index=s.index)

    df["rmssd_ms"] = df.groupby("operator_id")["heart_rate_bpm"].transform(rolling_rmssd)
    rmssd_min, rmssd_max = 10.0, 80.0  # RMSSD normalization range
    df["hrv_score"] = ((df["rmssd_ms"] - rmssd_min) / (rmssd_max - rmssd_min) * 100.0).clip(0, 100)
    df["hr_penalty"] = (df["hr_z_abs"].clip(0, 3) / 3.0) * 40.0  # z-score absolute penalty
    df["score1"] = (df["hrv_score"] - df["hr_penalty"]).clip(0, 100)

    # ---- score2: Chat positivity/consistency ----
    # Use dictionary module for keyword scoring
    print("Initializing keyword dictionary scorer...")
    dictionary_scorer = DictionaryScorer()
    
    # Show available category info
    print("Available keyword categories:")
    for category in dictionary_scorer.get_categories():
        info = dictionary_scorer.get_category_info(category)
        print(f"  - {category}: score delta {info['score_delta']}, keyword count {len(info['keywords'])}")
    print()
    
    def chat_inst_score(msg: str) -> float:
        """
        Use dictionary module for single chat scoring.
        Base score 60, adjusted by keyword match results.
        """
        if msg is None or msg == "nan":
            return np.nan
        
        # Use dictionary module to score
        result = dictionary_scorer.score_text(msg)
        score_delta = result.get('score_delta', 0)
        matched_keywords = result.get('matched_keywords', [])
        
        # Base score 60, adjust by score_delta
        base_score = 60.0
        score = base_score + score_delta * 10  # Amplify score_delta for more impact
        
        # If there are matched keywords, further adjust
        if matched_keywords:
            # Fine-tune by number of matches
            keyword_bonus = len(matched_keywords) * 2
            score += keyword_bonus
        
        return float(np.clip(score, 0, 100))

    # Demo scoring for some messages
    print("Keyword scoring demo:")
    demo_messages = ["fail!", "looks fine", "not sure", "urgent", "all good", "error!", "smooth"]
    for msg in demo_messages:
        result = dictionary_scorer.score_text(msg)
        score = chat_inst_score(msg)
        print(f"  '{msg}' -> category: {result['category']}, score delta: {result['score_delta']}, final score: {score:.1f}")
    print()
    
    df["chat_inst"] = df["message"].apply(chat_inst_score)

    # Momentum smoothing (exponential decay), slowly return to 60 if no message
    df["score2"] = np.nan
    for op in ops:
        mask = df["operator_id"] == op
        series = df.loc[mask, "chat_inst"].tolist()
        out_vals = []
        prev = 60.0
        alpha = 0.3
        decay_no_msg = 0.98
        for val in series:
            if not np.isnan(val):
                prev = (1 - alpha) * prev + alpha * val
            else:
                prev = prev * decay_no_msg + (1 - decay_no_msg) * 60.0
            out_vals.append(max(0.0, min(100.0, prev)))
        df.loc[mask, "score2"] = out_vals

    # ---- score3: Task health ----
    if not tasks.empty:
        bt = assign_bins(tasks, bin_seconds)
        df["score3"] = np.nan
        for op in ops:
            op_events = bt[bt["operator_id"] == op].sort_values("timestamp")
            # Count task start/complete/error per bin (ceil to bin boundary to ensure causality)
            per_bin = op_events.groupby(op_events["timestamp"].dt.ceil(f"{bin_seconds}s")).agg(
                starts=("event", lambda s: (s == "task start").sum()),
                completes=("event", lambda s: (s == "task complete").sum()),
                errors=("event", lambda s: (s == "task error").sum()),
            ).reset_index().rename(columns={"timestamp": "bin_ts"})
            print('per_bin', per_bin)
            mask = df["operator_id"] == op
            op_df = df.loc[mask, ["bin_ts"]].copy()
            op_df = op_df.merge(per_bin, on="bin_ts", how="left").fillna(0)
            print('op_df', op_df)
            # Rolling window statistics
            # rolling(window=6, min_periods=1) means: the first point has only itself, the second has the first two, ... from the 6th and after, always 6 (including itself), i.e. each point looks back at most 6 (including itself) for cumulative sum
            starts_roll = op_df["starts"].rolling(window=6, min_periods=1).sum()
            completes_roll = op_df["completes"].rolling(window=6, min_periods=1).sum()
            errors_roll = op_df["errors"].rolling(window=6, min_periods=1).sum()
            print('op_df', op_df)
            # Print full rolling statistics
            # Output full rolling statistics table
            roll_df = op_df.copy()
            roll_df["starts_roll"] = starts_roll
            roll_df["completes_roll"] = completes_roll
            roll_df["errors_roll"] = errors_roll
            print(f"\nOperator {op} rolling statistics table:")
            print(roll_df.to_string(index=False))
            # Calculate accuracy (completion rate) and error rate
            with np.errstate(divide="ignore", invalid="ignore"):
                # If starts_roll==0, comp_rate default 0.5, err_rate default 0.0
                comp_rate = np.where(starts_roll > 0, completes_roll / starts_roll, 0.5)
                err_rate = np.where(starts_roll > 0, errors_roll / starts_roll, 0.0)

            # Calculate average task starts (average number of starts per bin in window)
            # Note: window is small at first, denominator can't be 0
            window_sizes = np.arange(1, len(starts_roll) + 1)
            avg_starts = starts_roll / window_sizes

            # Task health score calculation
            # 1. Higher completion rate is better, lower error rate is better, average starts per bin moderate (too high/low penalized)
            # 2. Completion rate is main, error rate negative, average starts deviation from 2 (per bin) penalized
            # 3. Score range 0-100
            base = 100.0 * comp_rate

            # Average starts penalty (ideal 4 per bin, deviation penalized, max 20 points)
            ideal_starts = 4.0
            # Penalty is |avg_starts - 4| * 8, max 20
            start_penalty = np.minimum(np.abs(avg_starts - ideal_starts) * 8.0, 20.0)

            inst = np.clip(base - start_penalty, 0, 100)

            # Momentum smoothing
            out_vals = []
            prev = 70.0
            alpha = 0.15
            for val in inst:
                prev = (1 - alpha) * prev + alpha * val
                out_vals.append(max(0.0, min(100.0, prev)))
            df.loc[mask, "score3"] = out_vals

    else:
        # Default 60 if no task data
        df["score3"] = 60.0

    df["score1"] = (df["score1"]-30)*20/18+50
    df["score2"] = (df["score2"]-66)*20/3.5+50
    df["score3"] = (df["score3"]-50)*20/15+50

    # Clip scores to 0~100
    df["score1"] = df["score1"].clip(0, 100)
    df["score2"] = df["score2"].clip(0, 100)
    df["score3"] = df["score3"].clip(0, 100)

    # ---- Weighted total score ----
    w1, w2, w3 = weights
    df["score_all"] = (w1 * df["score1"].fillna(60.0) +
                       w2 * df["score2"].fillna(60.0) +
                       w3 * df["score3"].fillna(60.0))

    # ---- Generate comments ----
    df = df.sort_values(["bin_ts", "operator_id"]).reset_index(drop=True)
    df["comments"] = ""

    def hist_stats(arr: List[float]) -> Tuple[float, float]:
        """
        Calculate historical mean and std (ignore nan, sample std ddof=0).
        """
        a = np.array(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size < 2:
            return np.nan, np.nan
        return float(a.mean()), float(a.std(ddof=0))

    # Historical score cache
    hist_all: List[float] = []
    hist_s1: List[float] = []
    hist_s2: List[float] = []
    hist_s3: List[float] = []
    comments_out = []
    for _, row in df.iterrows():
        cur_all, cur_s1, cur_s2, cur_s3 = row["score_all"], row["score1"], row["score2"], row["score3"]
        mean_all, std_all = 50,20
        mean_s1, std_s1 = 50,20
        mean_s2, std_s2 = 50,20
        mean_s3, std_s3 = 50,20

        parts = []
        # If total score < historical mean-1σ, check if any of the three scores < its class mean-1σ, if so, write reason in comments
        if not np.isnan(mean_all) and not np.isnan(std_all) and cur_all < (mean_all - std_all):
            if (not np.isnan(mean_s1) and not np.isnan(std_s1) and not np.isnan(cur_s1) and cur_s1 < (mean_s1 - std_s1)):
                reason = "Low Score1: HRV (RMSSD) is low or heart rate fluctuation is abnormal, possible physiological stress/fatigue."
                ctx = []
                if not pd.isna(row.get("rmssd_ms", np.nan)): ctx.append(f"RMSSD≈{row['rmssd_ms']:.1f}ms")
                if not pd.isna(row.get("hr_z_abs", np.nan)): ctx.append(f"|z|≈{row['hr_z_abs']:.2f}")
                if not pd.isna(row.get("heart_rate_bpm", np.nan)): ctx.append(f"HR≈{row['heart_rate_bpm']:.1f}bpm")
                if ctx: reason += " (" + ", ".join(ctx) + ")"
                parts.append(reason)
            if (not np.isnan(mean_s2) and not np.isnan(std_s2) and not np.isnan(cur_s2) and cur_s2 < (mean_s2 - std_s2)):
                reason = "Low Score2: Too many negative/stress words or insufficient positive feedback in chat, possible communication disorder."
                msg = row.get("message", "")
                if isinstance(msg, str) and msg: reason += f" (Excerpt: {msg[:40]})"
                parts.append(reason)
            if (not np.isnan(mean_s3) and not np.isnan(std_s3) and not np.isnan(cur_s3) and cur_s3 < (mean_s3 - std_s3)):
                reason = "Low Score3: Task error rate increased/completion rate decreased/cycle time prolonged, possible rework or blockage."
                ev = row.get("events", "")
                if isinstance(ev, str) and ev: reason += f" (Events: {ev[:40]})"
                parts.append(reason)

        comments_out.append("; ".join(parts))
        # Update history
        hist_all.append(cur_all); hist_s1.append(cur_s1); hist_s2.append(cur_s2); hist_s3.append(cur_s3)

    df["comments"] = comments_out

    # Adjust output column order: message, events, comments to the end
    out_cols = ["bin_ts", "operator_id", "heart_rate_bpm", "score1", "score2", "score3", "score_all", "message", "events", "comments"]
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan
    out = df[out_cols].rename(columns={"bin_ts": "timestamp"})
    # Remove +00:00 at the end of timestamp
    if "timestamp" in out.columns:
        out["timestamp"] = out["timestamp"].astype(str).str.replace(r"\+00:00$", "", regex=True)

    # Ensure output has at most 2 decimal places
    float_cols = ["heart_rate_bpm", "score1", "score2", "score3", "score_all"]
    for col in float_cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: round(x, 2) if pd.notnull(x) else x)

    return out

def main():
    """
    Main entry: load data, compute scores, output csv.
    """
    hr, chat, tasks = load_inputs(BASE)
    timeline, ops = make_time_bins(hr, chat, tasks, bin_seconds=10)
    if not timeline or not ops:
        print("No valid data in inputs.")
        return
    out = compute_scores(hr, chat, tasks, timeline, ops, bin_seconds=10, weights=(0.34, 0.33, 0.33))
    out_path = BASE / "all.csv"
    out.to_csv(out_path, index=False, float_format="%.2f")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
