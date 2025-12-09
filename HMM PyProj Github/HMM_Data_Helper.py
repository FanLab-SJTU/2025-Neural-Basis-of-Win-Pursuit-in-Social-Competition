import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# ===================== 1. 数据导入 =====================
def load_daily_sequence(df, animal_id, test_day):
    """
    读取某一动物在特定 test_day 的 win/go 序列，返回 0/1 列向量 (T,1)
    1 = win side, 0 = go side
    """
    df_day = df[(df["ID"] == animal_id) & (df["Test_Day"] == test_day)].copy()
    if df_day.empty:
        return np.array([]).reshape(-1, 1)

    trial_columns = [f"Trial_{i}" for i in range(1, 51)]
    day_sequence = []

    for _, row in df_day.iterrows():
        win_side = str(row["Win_Side"]).strip()
        for col in trial_columns:
            if col in row and pd.notnull(row[col]):
                choice = str(row[col]).strip().upper()
                if choice in ["R", "L"]:
                    day_sequence.append(1 if choice == win_side else 0)

    return np.array(day_sequence, dtype=int).reshape(-1, 1)


def get_all_animals(csv_path):
    """
    返回:
      all_animals: {animal_id: {"train": train_seq_1to4, "days": {day: seq}}}
    不再要求 0-4 天完整；train 只用 win days 1-4，有多少用多少。
    baseline day=0 仅用于 posterior 计算。
    """
    df = pd.read_csv(csv_path)
    df["ID"] = df["ID"].astype(str).str.strip()
    df["Win_Side"] = df["Win_Side"].astype(str).str.strip()
    df["Test_Day"] = pd.to_numeric(df["Test_Day"], errors="coerce")

    all_animals = {}
    for aid in sorted(df["ID"].unique()):
        day_seqs = {}
        for day in [0, 1, 2, 3, 4]:
            seq = load_daily_sequence(df, aid, day)
            day_seqs[day] = seq

        # 拼接 win days 1-4 作为该动物的训练序列
        train_segments = [day_seqs[d] for d in [1, 2, 3, 4] if day_seqs[d].size > 0]
        if len(train_segments) > 0:
            train_seq = np.vstack(train_segments)
        else:
            train_seq = np.array([], dtype=int).reshape(-1, 1)

        all_animals[aid] = {"train": train_seq, "days": day_seqs}

    return all_animals

def build_pooled_train(all_animals):
    segments = []
    lengths = []
    used_animals = []

    for aid, data in all_animals.items():
        seq = data["train"]
        if seq.size == 0:
            continue
        segments.append(seq)
        lengths.append(len(seq))
        used_animals.append(aid)

    if len(segments) == 0:
        raise ValueError("No training sequences found for pooled HMM.")

    pooled_seq = np.vstack(segments)
    return pooled_seq, lengths, used_animals

def export_model_params_to_csv(model, state_names, save_path):
    rows = []
    K = model.n_components

    # startprob
    for s in range(K):
        rows.append(dict(
            matrix="startprob",
            from_state=s,
            to_state=None,
            obs=None,
            state_name=state_names[s],
            value=model.startprob_[s]
        ))

    # transmat
    for i in range(K):
        for j in range(K):
            rows.append(dict(
                matrix="transmat",
                from_state=i,
                to_state=j,
                obs=None,
                state_name=state_names[i],
                to_state_name=state_names[j],
                value=model.transmat_[i, j]
            ))

    # emission prob, obs=0(go) / 1(win)
    obs_labels = {0: "go(0)", 1: "win(1)"}
    for i in range(K):
        for o in [0, 1]:
            rows.append(dict(
                matrix="emission",
                from_state=i,
                to_state=None,
                obs=o,
                state_name=state_names[i],
                obs_label=obs_labels[o],
                value=model.emissionprob_[i, o]
            ))

    df_param = pd.DataFrame(rows)
    df_param.to_csv(save_path, index=False)
    print(f"Model parameters saved -> {save_path}")
    
def smooth_posteriors(posteriors, sigma=1.2):
    sm = gaussian_filter1d(posteriors, sigma=sigma, axis=0, mode='nearest')
    sm = sm / sm.sum(axis=1, keepdims=True)
    return sm