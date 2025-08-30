
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
all.py — 统一整合 1run/2run/3run 三类信号，每10秒评估一次并输出加权总评分。
输出: all.csv，包含
timestamp, operator_id, heart_rate_bpm, score1, score2, score3, score_all, message, events, comments

说明：
- score1（0-100，越高越好）：基于心率RMSSD（HRV）+ 心率z分异常惩罚
- score2（0-100，越高越好）：聊天积极度/一致性（关键词 + 动量平滑）
- score3（0-100，越高越好）：任务健康度（完结率↑、错误率↓、周期时间↓ + 动量平滑）
- score_all：加权和（默认 0.34/0.33/0.33）

若当前某点的 score_all 低于历史（全部人全部时间，严格到当前之前）均值 - 1σ，
则检查三项分数中是否有任一项 < 其“本类历史均值 - 1σ”（全部人全部时间，严格到当前之前，且只在本类内部比较）。
若有，则在 comments 中给出原因描述和少量上下文。
"""

from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
from datetime import timezone

import sys

class TeeLogger:
    """
    TeeLogger: 同时将print内容输出到stdout和all/all.log
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

# 先清空日志文件，再重定向stdout
with open("all/all.log", "w", encoding="utf-8"):
    pass  # 仅用于清空内容
sys.stdout = TeeLogger("all/all.log")

BASE = Path(".")

def read_csv_safely(path: Path) -> pd.DataFrame:
    """
    安全读取csv文件，若不存在则返回空DataFrame，兼容不同pandas版本的错误参数。
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(path, error_bad_lines=False)

def load_inputs(base: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    读取三类输入数据（心率、聊天、任务），并做基础字段校验和时间戳转换。
    """
    hr = read_csv_safely(base / "all/1input.csv")
    chat = read_csv_safely(base / "all/2input.csv")
    tasks = read_csv_safely(base / "all/3input.csv")
    # 转换时间戳为UTC
    for df in [hr, chat, tasks]:
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    # 校验字段，若缺失则补空表
    if not {"timestamp", "operator_id", "heart_rate_bpm"}.issubset(set(hr.columns)):
        hr = pd.DataFrame(columns=["timestamp", "operator_id", "heart_rate_bpm"])
    if not {"timestamp", "operator_id", "message"}.issubset(set(chat.columns)):
        chat = pd.DataFrame(columns=["timestamp", "operator_id", "message"])
    if not {"timestamp", "operator_id", "event", "task_id"}.issubset(set(tasks.columns)):
        tasks = pd.DataFrame(columns=["timestamp", "operator_id", "event", "task_id"])
    return hr, chat, tasks

def make_time_bins(hr: pd.DataFrame, chat: pd.DataFrame, tasks: pd.DataFrame, bin_seconds: int = 10) -> Tuple[List[pd.Timestamp], List[str]]:
    """
    计算全体数据的时间范围，生成每bin_seconds一格的时间轴，以及所有出现过的operator_id列表。
    """
    dfs = [df for df in [hr, chat, tasks] if not df.empty]
    if not dfs:
        return [], []
    # 取所有数据的最小/最大时间
    tmin = min(df["timestamp"].min() for df in dfs if not df["timestamp"].isna().all())
    tmax = max(df["timestamp"].max() for df in dfs if not df["timestamp"].isna().all())
    print('tmin, tmax', tmin, tmax)
    if pd.isna(tmin) or pd.isna(tmax):
        return [], []
    # 对齐到bin边界
    tmin = (tmin - pd.Timedelta(seconds=tmin.second % bin_seconds, microseconds=tmin.microsecond)).floor(f"{bin_seconds}s")
    tmax = (tmax + pd.Timedelta(seconds=(bin_seconds - (tmax.second % bin_seconds)) % bin_seconds)).ceil(f"{bin_seconds}s")
    timeline = list(pd.date_range(tmin, tmax, freq=f"{bin_seconds}s", tz=timezone.utc))
    # 汇总所有operator_id
    ops = sorted(set(hr.get("operator_id", pd.Series(dtype=str)).dropna().unique().tolist()
                     + chat.get("operator_id", pd.Series(dtype=str)).dropna().unique().tolist()
                     + tasks.get("operator_id", pd.Series(dtype=str)).dropna().unique().tolist()))

    print('timeline', timeline)
    print('ops', ops)
    return timeline, ops

def assign_bins(df: pd.DataFrame, bin_seconds: int = 10) -> pd.DataFrame:
    """
    给每条数据分配所属的时间bin（向下取整）。
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
    主评分计算流程：聚合数据、计算三项分数、加权总分、生成comments。
    """
    # 1. 按bin聚合三类数据
    bhr = assign_bins(hr, bin_seconds)
    bchat = assign_bins(chat, bin_seconds)
    btasks = assign_bins(tasks, bin_seconds)
    # 心率聚合：每bin取均值
    hr_agg = bhr.groupby(["bin_ts", "operator_id"], as_index=False).agg(
        heart_rate_bpm=("heart_rate_bpm", "mean")
    )

    # 这里将聊天数据按bin_ts和operator_id聚合，每个bin内拼接所有消息（用" | "分隔），但只包含bin_ts这个时刻及之前的原始数据
    # 例如 2025-08-20 10:00:10+00:00 这个bin，只包含时间戳小于等于10:00:10的原始数据
    # 这样每个bin只反映该时刻及之前的内容，不包含之后的
    # 修复聊天聚合部分
    chat_agg = []
    chat_by_op = chat.groupby("operator_id")
    for bin_ts in timeline:
        for op in ops:
            if op in chat_by_op.groups:
                op_chat = chat_by_op.get_group(op)
                # 修复：正确的时间范围筛选
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
            
                

    # 任务聚合：每bin拼接事件:task_id
    # 任务聚合：每bin拼接事件:task_id，方式与chat_agg一致，确保每个bin只包含该时刻及之前的事件
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

    # 生成完整的时间-operator网格，左连接三类聚合数据
    grid = pd.MultiIndex.from_product([timeline, ops], names=["bin_ts", "operator_id"]).to_frame(index=False)
    df = grid.merge(hr_agg, on=["bin_ts", "operator_id"], how="left") \
             .merge(chat_agg, on=["bin_ts", "operator_id"], how="left") \
             .merge(task_agg, on=["bin_ts", "operator_id"], how="left")

    window_bins = 6  # 滚动窗口长度（6*10s=1分钟）
    df = df.sort_values(["operator_id", "bin_ts"]).reset_index(drop=True)

    # ---- score1: HRV+心率z分惩罚 ----
    # 计算滚动均值/方差/z分（只参考当前及之前的值，不包含未来）
    # 注意：rolling默认window是右闭区间（包含当前，不包含未来），center=False
    df["hr_roll_mean"] = df.groupby("operator_id")["heart_rate_bpm"].transform(
        lambda s: s.rolling(window=window_bins, min_periods=3).mean()
    )
    df["hr_roll_std"] = df.groupby("operator_id")["heart_rate_bpm"].transform(
        lambda s: s.rolling(window=window_bins, min_periods=3).std()
    )
    df["hr_z"] = (df["heart_rate_bpm"] - df["hr_roll_mean"]) / df["hr_roll_std"]
    df.loc[df["hr_roll_std"].isna() | (df["hr_roll_std"] == 0), "hr_z"] = np.nan
    df["hr_z_abs"] = df["hr_z"].abs()

    # 计算滚动RMSSD（同样只用当前及之前的值）
    def rmssd_from_hr(series_bpm: pd.Series) -> float:
        """
        计算一组心率（bpm）的RMSSD（HRV指标）。
        """
        hr = series_bpm.dropna().astype(float)
        if len(hr) < 3:
            return np.nan
        rr = 60000.0 / hr  # RR间期（ms）
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
    rmssd_min, rmssd_max = 10.0, 80.0  # RMSSD归一化区间
    df["hrv_score"] = ((df["rmssd_ms"] - rmssd_min) / (rmssd_max - rmssd_min) * 100.0).clip(0, 100)
    df["hr_penalty"] = (df["hr_z_abs"].clip(0, 3) / 3.0) * 40.0  # z分绝对值惩罚
    df["score1"] = (df["hrv_score"] - df["hr_penalty"]).clip(0, 100)

    # ---- score2: 聊天积极度/一致性 ----
    # 关键词词典
    # 负向和正向关键词各扩充到约50个，便于更细致地评估聊天情绪
    stress_words = {
        "error", "fail", "failed", "issue", "stuck", "delay", "urgent", "can't", "cannot", "wtf", "broken", "retry", "oops", "panic",
        "problem", "trouble", "difficult", "hard", "slow", "blocked", "miss", "lost", "down", "crash", "freeze", "hang", "stop",
        "unavailable", "unresponsive", "disconnect", "timeout", "overload", "overheat", "warning", "alert", "critical", "fatal",
        "danger", "risk", "unstable", "unexpected", "abnormal", "corrupt", "invalid", "denied", "refused", "rejected", "conflict",
        "collision", "late", "postpone", "stress", "tired", "exhausted", "annoy", "frustrate", "angry", "upset", "disappointed",
        "sad", "complain", "regret", "sorry", "pain", "hurt", "worry", "afraid", "fear", "scared", "terrified", "confuse", "mess",
        "chaos", "disaster", "unlucky", "unfortunate", "unable", "incomplete", "unhappy", "hopeless", "helpless", "useless",
        "pointless", "meaningless", "worthless", "bug", "glitch", "lag", "dead", "jam", "sucks", "hate", "dislike", "disgust"
    }
    positive_words = {
        "ok", "done", "ready", "roger", "nice", "great", "thanks", "thank you", "proceed", "confirm", "confirmed", "noted", "on it",
        "looks fine", "received", "good", "well done", "perfect", "excellent", "awesome", "fine", "clear", "all good", "smooth",
        "success", "successful", "fixed", "solved", "resolved", "stable", "fast", "faster", "quick", "quickly", "improve", "improved",
        "improving", "improvement", "safe", "safely", "safety", "reliable", "reliably", "trust", "trustworthy", "cooperate",
        "cooperation", "help", "helpful", "productive", "motivated", "enthusiastic", "engaged", "committed", "willing", "coordinated",
        "organized", "structured", "planned", "prepared", "ready", "available", "responsive", "usable", "workable", "manageable",
        "controllable", "predictable", "support", "supported", "supporting", "supportive", "cheer", "cheers", "happy", "happiness",
        "enjoy", "enjoyed", "enjoying", "enjoys", "like", "liked", "liking", "likes", "love", "loved", "loving", "loves"
    }
    df["message_lc"] = df["message"].astype(str).str.lower()

    def chat_inst_score(msg: str) -> float:
        """
        单bin聊天分数：基础60，正向词+8，负向词-10，限0-100
        """
        if msg is None or msg == "nan":
            return np.nan
        score = 60.0
        s_cnt = sum(1 for w in stress_words if w in msg)
        p_cnt = sum(1 for w in positive_words if w in msg)
        score += 8.0 * p_cnt
        score -= 10.0 * s_cnt
        return float(np.clip(score, 0, 100))

    df["chat_inst"] = df["message_lc"].apply(chat_inst_score)

    # 动量平滑（指数衰减），无消息时缓慢回归60
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

    # ---- score3: 任务健康度 ----
    if not tasks.empty:
        bt = assign_bins(tasks, bin_seconds)
        df["score3"] = np.nan
        for op in ops:
            op_events = bt[bt["operator_id"] == op].sort_values("timestamp")
            # 每bin统计任务开始/完成/错误数（向上取整到bin边界，确保因果性）
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
            # 滚动窗口统计
            # 是的，rolling(window=6, min_periods=1)的意思是：最开始第1个点窗口里只有1个（自己），第2个点窗口里有前2个……直到第6个及以后，窗口里都是前6个（含自己），即每个点向前最多看6个（含自己）做累计和
            starts_roll = op_df["starts"].rolling(window=6, min_periods=1).sum()
            completes_roll = op_df["completes"].rolling(window=6, min_periods=1).sum()
            errors_roll = op_df["errors"].rolling(window=6, min_periods=1).sum()
            print('op_df', op_df)
            # 打印完整的滚动统计结果
            # 输出完整的滚动统计表格
            roll_df = op_df.copy()
            roll_df["starts_roll"] = starts_roll
            roll_df["completes_roll"] = completes_roll
            roll_df["errors_roll"] = errors_roll
            print(f"\n操作员 {op} 的滚动统计表：")
            print(roll_df.to_string(index=False))
            # with np.errstate(divide="ignore", invalid="ignore"):
            #     comp_rate = np.where(starts_roll > 0, completes_roll / starts_roll, 0.5)
            #     err_rate = np.where(starts_roll > 0, errors_roll / starts_roll, 0.0)
            # INSERT_YOUR_CODE
            # 计算准确率（完成率）和错误率
            with np.errstate(divide="ignore", invalid="ignore"):
                # starts_roll为0时，comp_rate默认0.5，err_rate默认0.0
                comp_rate = np.where(starts_roll > 0, completes_roll / starts_roll, 0.5)
                err_rate = np.where(starts_roll > 0, errors_roll / starts_roll, 0.0)

            # 计算平均开始任务数（窗口内平均每个bin有多少个start）
            # 注意最开始窗口很小，分母不能为0
            window_sizes = np.arange(1, len(starts_roll) + 1)
            avg_starts = starts_roll / window_sizes

            # 任务健康度分数计算
            # 1. 完成率越高越好，错误率越低越好，平均开始任务数适中（过高或过低都扣分）
            # 2. 以完成率为主，错误率为负向，平均开始任务数偏离2（每bin）时扣分
            # 3. 分数范围0-100
            base = 100.0 * comp_rate

            # 平均开始任务数惩罚（假设理想每bin 4个start，偏离4越多扣分越多，最大惩罚20分）
            ideal_starts = 4.0
            # 惩罚为 |avg_starts - 2| * 4，最大不超过20分
            start_penalty = np.minimum(np.abs(avg_starts - ideal_starts) * 8.0, 20.0)

            inst = np.clip(base - start_penalty, 0, 100)

            # 动量平滑
            out_vals = []
            prev = 70.0
            alpha = 0.15
            for val in inst:
                prev = (1 - alpha) * prev + alpha * val
                out_vals.append(max(0.0, min(100.0, prev)))
            df.loc[mask, "score3"] = out_vals


            # # 计算平均周期时间
            # start_times = {}
            # durations = []
            # for row in op_events.itertuples(index=False):
            #     if row.event == "task start":
            #         start_times[row.task_id] = row.timestamp
            #     elif row.event == "task complete" and row.task_id in start_times:
            #         durations.append((row.timestamp - start_times[row.task_id]).total_seconds())
            # avg_ct = float(np.mean(durations)) if durations else np.nan
            # base = 100.0 * comp_rate - 80.0 * err_rate
            # print('avg_ct', avg_ct)
            # # 周期时间惩罚
            # if not np.isnan(avg_ct):
            #     ct_penalty = np.interp(avg_ct, [15.0, 60.0, 120.0], [0.0, 15.0, 30.0])
            # else:
            #     ct_penalty = 0.0
            # print('ct_penalty', ct_penalty)
            # inst = np.clip(base - ct_penalty, 0, 100)
            # print('inst', inst)
            # # 动量平滑
            # out_vals = []
            # prev = 70.0
            # alpha = 0.4
            # for val in inst:
            #     prev = (1 - alpha) * prev + alpha * val
            #     out_vals.append(max(0.0, min(100.0, prev)))
            # df.loc[mask, "score3"] = out_vals
    else:
        # 无任务数据时默认70分
        df["score3"] = 60.0

    df["score1"] = (df["score1"]-30)*20/18+50
    df["score2"] = (df["score2"]-66)*20/3.5+50
    df["score3"] = (df["score3"]-50)*20/15+50

    # 分数裁剪到0~100
    df["score1"] = df["score1"].clip(0, 100)
    df["score2"] = df["score2"].clip(0, 100)
    df["score3"] = df["score3"].clip(0, 100)

    # ---- 总分加权 ----
    w1, w2, w3 = weights
    df["score_all"] = (w1 * df["score1"].fillna(60.0) +
                       w2 * df["score2"].fillna(60.0) +
                       w3 * df["score3"].fillna(60.0))

    # ---- comments生成 ----
    df = df.sort_values(["bin_ts", "operator_id"]).reset_index(drop=True)
    df["comments"] = ""

    def hist_stats(arr: List[float]) -> Tuple[float, float]:
        """
        计算历史均值和标准差（忽略nan，样本标准差ddof=0）。
        """
        a = np.array(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size < 2:
            return np.nan, np.nan
        return float(a.mean()), float(a.std(ddof=0))

    # 历史分数缓存
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
        # 总分低于历史均值-1σ时，检查三项分数是否有低于本类均值-1σ，若有则写入comments
        if not np.isnan(mean_all) and not np.isnan(std_all) and cur_all < (mean_all - std_all):
            if (not np.isnan(mean_s1) and not np.isnan(std_s1) and not np.isnan(cur_s1) and cur_s1 < (mean_s1 - std_s1)):
                reason = "Score1偏低：HRV（RMSSD）偏低或心率波动异常，可能生理压力/疲劳。"
                ctx = []
                if not pd.isna(row.get("rmssd_ms", np.nan)): ctx.append(f"RMSSD≈{row['rmssd_ms']:.1f}ms")
                if not pd.isna(row.get("hr_z_abs", np.nan)): ctx.append(f"|z|≈{row['hr_z_abs']:.2f}")
                if not pd.isna(row.get("heart_rate_bpm", np.nan)): ctx.append(f"HR≈{row['heart_rate_bpm']:.1f}bpm")
                if ctx: reason += "（" + "，".join(ctx) + "）"
                parts.append(reason)
            if (not np.isnan(mean_s2) and not np.isnan(std_s2) and not np.isnan(cur_s2) and cur_s2 < (mean_s2 - std_s2)):
                reason = "Score2偏低：聊天中负向/紧张词汇较多或积极反馈不足，沟通节奏可能紊乱。"
                msg = row.get("message", "")
                if isinstance(msg, str) and msg: reason += f"（片段：{msg[:40]}）"
                parts.append(reason)
            if (not np.isnan(mean_s3) and not np.isnan(std_s3) and not np.isnan(cur_s3) and cur_s3 < (mean_s3 - std_s3)):
                reason = "Score3偏低：任务错误率上升/完成率下降/周期时间偏长，存在返工或阻塞。"
                ev = row.get("events", "")
                if isinstance(ev, str) and ev: reason += f"（事件：{ev[:40]}）"
                parts.append(reason)

        comments_out.append("; ".join(parts))
        # 更新历史
        hist_all.append(cur_all); hist_s1.append(cur_s1); hist_s2.append(cur_s2); hist_s3.append(cur_s3)

    df["comments"] = comments_out

    # 调整输出列顺序：message, events, comments 移到最后
    out_cols = ["bin_ts", "operator_id", "heart_rate_bpm", "score1", "score2", "score3", "score_all", "message", "events", "comments"]
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan
    out = df[out_cols].rename(columns={"bin_ts": "timestamp"})
    # 去掉timestamp末尾的+00:00
    if "timestamp" in out.columns:
        out["timestamp"] = out["timestamp"].astype(str).str.replace(r"\+00:00$", "", regex=True)

    

    # 保证输出最多2位小数
    float_cols = ["heart_rate_bpm", "score1", "score2", "score3", "score_all"]
    for col in float_cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: round(x, 2) if pd.notnull(x) else x)

    return out

def main():
    """
    主入口：加载数据、计算分数、输出csv。
    """
    hr, chat, tasks = load_inputs(BASE)
    timeline, ops = make_time_bins(hr, chat, tasks, bin_seconds=10)
    if not timeline or not ops:
        print("No valid data in inputs.")
        return
    out = compute_scores(hr, chat, tasks, timeline, ops, bin_seconds=10, weights=(0.34, 0.33, 0.33))
    out_path = BASE / "all/all.csv"
    out.to_csv(out_path, index=False, float_format="%.2f")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
