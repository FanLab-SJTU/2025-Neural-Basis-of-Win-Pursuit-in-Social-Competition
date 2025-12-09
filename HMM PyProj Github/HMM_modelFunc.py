import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from hmmlearn.hmm import CategoricalHMM
from scipy.ndimage import gaussian_filter1d



def clamp_emission_semantic_3state(emissionprob,
                                   win_range=(0.80, 1.00),
                                   rand_range=(0.45, 0.55),
                                   go_range=(0.00, 0.20)):
    """
    假定 emissionprob 形状为 (3, 2)，行顺序已经是：
        0: Win-like
        1: Random-like
        2: Go-like
    每行是 [P(obs=0), P(obs=1)]，其中 obs=1 = win。

    将每行对应的 P(win) 强制限制在指定区间内。
    """
    em = emissionprob.copy()
    ranges = [win_range, rand_range, go_range]

    for row_idx, (lo, hi) in enumerate(ranges):
        # 当前 state 的 P(win)：
        p1 = em[row_idx, 1]
        # 限制在区间内
        p1 = np.clip(p1, lo, hi)
        em[row_idx, 1] = p1
        em[row_idx, 0] = 1.0 - p1  # 只有 0 和 1 两个观测，行和必须是 1

    return em

def shrink_rows_to_uniform(mat, lam=0.05):
    """
    对每一行做 soft regularization:
      mat_new = (1-lam)*mat + lam*uniform
    其中 uniform 是该行上的均匀分布。
    lam 越大，越靠近均匀；保证没有元素是 0 或 1。
    """
    k = mat.shape[1]
    uniform = np.ones_like(mat) / float(k)
    mat_new = (1.0 - lam) * mat + lam * uniform
    mat_new /= mat_new.sum(axis=1, keepdims=True)
    return mat_new


def mean_row_entropy(mat):
    """
    计算按行的平均熵，用来识别“极端”的转移矩阵 / 发射矩阵。
    熵越低，表示越极端（接近 one-hot）。
    """
    mat_safe = np.clip(mat, 1e-12, 1.0)
    row_ent = -(mat_safe * np.log(mat_safe)).sum(axis=1)
    return float(row_ent.mean())

# ===================== 3. 按发射到 win 的概率对 state 重排并保存参数 =====================
def reorder_states_by_win_emission(model):
    """
    按发射到 win(1) 的概率从高到低重排 state：
      （n_states == 3 时，定义：
        0 -> Win-like
        1 -> Random-like
        2 -> Go-like ）
    返回：
      new_model   : 重排后的 HMM
      state_names : 重排后每个 state 的名称列表
      order       : 原始 state 索引 -> 新顺序 的排列，如 [2,0,1]
    """
    emit = model.emissionprob_.copy()   # shape: (K, 2)
    p_win = emit[:, 1]                  # 每个 state 发射 win(1) 的概率

    # 按 P(win) 从大到小排序
    order = np.argsort(p_win)[::-1]

    start_new = model.startprob_[order]
    trans_new = model.transmat_[np.ix_(order, order)]
    emit_new  = emit[order, :]

    # 新建一个 HMM，把参数拷进去
    new_model = CategoricalHMM(
        n_components=model.n_components,
        n_iter=model.n_iter,
        random_state=model.random_state
    )
    new_model.n_features    = model.n_features
    new_model.startprob_    = start_new
    new_model.transmat_     = trans_new
    new_model.emissionprob_ = emit_new

    # state 命名：你目前用 3-state HMM，就用固定命名
    if model.n_components == 3:
        state_names = ["Win-like", "Random-like", "Go-like"]
    else:
        # 如果以后要跑别的 K，也不会报错
        state_names = [f"state_{i}" for i in range(model.n_components)]

    return new_model, state_names, order

# ===================== 2. pooled win days 训练 3-state HMM =====================
def fit_stable_avg_hmm(train_seq, lengths, n_states,
                       n_rounds=10,
                       n_init_per_round=10,
                       n_iter=300,
                       random_seed=666,
                       z_cut=3.0,
                       # lam_trans=0.05,
                       entropy_min_trans=0.15,
                       entropy_min_emit=0.15
                       ):

    rng = np.random.RandomState(random_seed)

    start_list = []
    trans_list = []
    emit_list  = []
    all_lls    = []
    trans_ent_list = []
    emit_ent_list  = []

    for r in range(n_rounds):
        print(f"[Round {r+1}/{n_rounds}]")

        best_model_r = None
        best_ll_r = -np.inf
        best_trans_ent_r = None
        best_emit_ent_r  = None

        for i in range(n_init_per_round):
            rs = int(rng.randint(0, 1e9))
            model = CategoricalHMM(
                n_components=n_states,
                n_iter=n_iter,
                random_state=rs,
                verbose=False,
            )
            model.n_features = 2  # obs={0,1}
            model.fit(train_seq, lengths=lengths)

            # 先按 P(win) 排序，确保 row0/1/2 分别对齐 Win/Random/Go
            model_aligned, _, _ = reorder_states_by_win_emission(model)

            # # （可选）对转移矩阵做一点软正则，避免极端 0/1
            # model_aligned.transmat_ = shrink_rows_to_uniform(
            #     model_aligned.transmat_, lam=lam_trans
            # )

            # # 对 emission 做语义上的 clamp：
            # #   row0: win-like (P(win) 高)
            # #   row1: random-like (P(win) 中)
            # #   row2: go-like (P(win) 低)
            # model_aligned.emissionprob_ = clamp_emission_semantic_3state(
            #     model_aligned.emissionprob_,
            #     win_range=(0.80, 1.00),
            #     rand_range=(0.45, 0.55),
            #     go_range=(0.00, 0.20)
            # )

            # 用“排好序+clamp”的模型算 logL
            total_ll = 0.0
            idx = 0
            for L in lengths:
                seg = train_seq[idx:idx+L]
                total_ll += model_aligned.score(seg)
                idx += L

            if total_ll > best_ll_r:
                best_ll_r = total_ll
                best_model_r = model_aligned

        # 一轮结束：记录这一轮的 best（已经 aligned + clamped）
        trans_ent = mean_row_entropy(best_model_r.transmat_)
        emit_ent  = mean_row_entropy(best_model_r.emissionprob_)

        start_list.append(best_model_r.startprob_.copy())
        trans_list.append(best_model_r.transmat_.copy())
        emit_list.append(best_model_r.emissionprob_.copy())
        all_lls.append(best_ll_r)
        trans_ent_list.append(trans_ent)
        emit_ent_list.append(emit_ent)

    all_lls = np.array(all_lls)
    trans_ent_list = np.array(trans_ent_list)
    emit_ent_list  = np.array(emit_ent_list)

    print("Log-likelihood per round:", all_lls)             
    print("Mean transition entropy per round:", trans_ent_list)
    print("Mean emission entropy per round:",  emit_ent_list)

    # ---------- (1) 先在 logL 维度做 robust outlier 剔除 ----------
    median_ll = np.median(all_lls)
    mad = np.median(np.abs(all_lls - median_ll))

    if mad == 0:
        ll_mask = np.ones_like(all_lls, dtype=bool)
    else:
        robust_z = 0.6745 * (all_lls - median_ll) / mad
        ll_mask = robust_z > -z_cut   # 剔除明显更差的轮

    # ---------- (2) 再根据熵剔除“过于极端”的轮 ----------
    ent_mask = (trans_ent_list > entropy_min_trans) & (emit_ent_list > entropy_min_emit)

    keep_mask = ll_mask & ent_mask
    kept_idx = np.where(keep_mask)[0].tolist()
    print(f"Kept rounds after LL+entropy filtering: {kept_idx}")

    if keep_mask.sum() == 0:
        print("Warning: no rounds kept after filtering; fall back to global best.")
        best_global_idx = int(np.argmax(all_lls))
        keep_mask = np.zeros_like(all_lls, dtype=bool)
        keep_mask[best_global_idx] = True
        kept_idx = [best_global_idx]

    # ---------- (3) 对保留轮的参数求平均 ----------
    start_stack = np.stack(start_list, axis=0)[keep_mask]
    trans_stack = np.stack(trans_list, axis=0)[keep_mask]
    emit_stack  = np.stack(emit_list,  axis=0)[keep_mask]

    start_avg = np.mean(start_stack, axis=0)
    trans_avg = np.mean(trans_stack, axis=0)
    emit_avg  = np.mean(emit_stack,  axis=0)

    # 构造最终稳定模型
    stable_model = CategoricalHMM(
        n_components=n_states,
        n_iter=1,
        init_params="",
        params=""
    )
    stable_model.n_features    = 2
    stable_model.startprob_    = start_avg
    stable_model.transmat_     = trans_avg
    stable_model.emissionprob_ = emit_avg

    # 计算 stable 模型在训练数据上的 logL
    idx = 0
    stable_ll = 0.0
    for L in lengths:
        seg = train_seq[idx:idx+L]
        stable_ll += stable_model.score(seg)
        idx += L

    print(f"Stable-avg model logL = {stable_ll:.3f}, "
          f"(mean LL of kept rounds = {all_lls[keep_mask].mean():.3f})")

    return stable_model, stable_ll, kept_idx, all_lls


def fit_pooled_hmm_fixed_emission(train_seq, lengths,
                                  emission_fixed,
                                  n_states=3,
                                  n_init=20,
                                  n_iter=300,
                                  random_seed=0):
    """
    使用 hmmlearn 的 CategoricalHMM, 但:
      - 预先固定 emissionprob_ = emission_fixed
      - EM 只拟合 startprob_ + transmat_ (params="st")
    多次随机初始化, 选训练 log-likelihood 最高的模型。
    """
    rng = np.random.RandomState(random_seed)
    best_model = None
    best_ll = -np.inf

    for k in range(n_init):
        rs = int(rng.randint(0, 1e9))

        model = CategoricalHMM(
            n_components=n_states,
            n_iter=n_iter,
            random_state=rs,
            verbose=False,
            # 只更新 startprob(s) & transmat(t), 不更新 emission(e)
            params="st",
            init_params="st"
        )
        model.n_features = 2  # obs in {0,1}

        # 固定 emission，不让 HMM 动它
        model.emissionprob_ = emission_fixed.copy()

        # 拟合 pooled 序列
        model.fit(train_seq, lengths=lengths)

        # 精确算训练 log-likelihood (按每段相加)
        total_ll = 0.0
        idx = 0
        for L in lengths:
            seg = train_seq[idx:idx + L]
            total_ll += model.score(seg)
            idx += L

        if total_ll > best_ll:
            best_ll = total_ll
            best_model = model

    print(f"[fit_pooled_hmm_fixed_emission] best total logL = {best_ll:.3f}")
    return best_model, best_ll

# average model weight by LL
def fit_pooled_hmm_fixed_emission_avg(train_seq, lengths,
                                      emission_fixed,
                                      n_states=3,
                                      n_init=50,
                                      n_iter=300,
                                      random_seed=0,
                                      top_frac=0.5):
    """
    使用 hmmlearn CategoricalHMM 拟合 pooled HMM:
      - 固定 emissionprob_ = emission_fixed
      - EM 只更新 startprob_ + transmat_ (params="st")
    进行 n_init 次随机初始化，得到一组 {start, trans, logL}，
    最后对其中 logL 较好的那一半 (top_frac) 做平均，得到稳定模型。

    返回:
      avg_model : 平均后的 HMM (emission 固定)
      all_lls   : 每次初始化的 total log-likelihood 数组
    """
    rng = np.random.RandomState(random_seed)

    start_list = []
    trans_list = []
    ll_list    = []

    for k in range(n_init):
        print(f"Round:{k}/{n_init}")
        rs = int(rng.randint(0, 1e9))

        model = CategoricalHMM(
            n_components=n_states,
            n_iter=n_iter,
            random_state=rs,
            verbose=False,
            params="st",      # 只拟合 s,t
            init_params="st"
        )
        model.n_features   = 2
        model.emissionprob_ = emission_fixed.copy()  # 固定 emission

        model.fit(train_seq, lengths=lengths)

        # 计算 pooled 上的 total logL
        total_ll = 0.0
        idx = 0
        for L in lengths:
            seg = train_seq[idx:idx+L]
            total_ll += model.score(seg)
            idx += L

        start_list.append(model.startprob_.copy())
        trans_list.append(model.transmat_.copy())
        ll_list.append(total_ll)

    ll_array = np.array(ll_list)
    print("All total logL:", ll_array)

    # ---- 选取 logL 较好的那一部分（比如 top 50%） ----
    n_keep = max(1, int(len(ll_array) * top_frac))
    keep_idx = np.argsort(ll_array)[-n_keep:]  # 从小到大排序，取最后 n_keep 个

    print("Kept inits (indices):", keep_idx)
    print("Kept logL:", ll_array[keep_idx])

    start_keep = np.stack(start_list, axis=0)[keep_idx]
    trans_keep = np.stack(trans_list, axis=0)[keep_idx]

    # ---- 可以做等权平均，也可以用 logL 作为权重 ----
    # 等权平均：
    start_avg = start_keep.mean(axis=0)
    trans_avg = trans_keep.mean(axis=0)

    # # 如果想用 logL 做权重，可以改成：
    # ll_keep = ll_array[keep_idx]
    # w = np.exp(ll_keep - ll_keep.max())  # softmax 前的一步
    # w /= w.sum()
    # start_avg = np.tensordot(w, start_keep, axes=(0, 0))
    # trans_avg = np.tensordot(w, trans_keep, axes=(0, 0))

    # 保证归一化（数值稳定）
    start_avg /= start_avg.sum()
    trans_avg /= trans_avg.sum(axis=1, keepdims=True)

    # 构建平均模型
    avg_model = CategoricalHMM(
        n_components=n_states,
        n_iter=1,
        init_params="",
        params=""
    )
    avg_model.n_features    = 2
    avg_model.emissionprob_ = emission_fixed.copy()
    avg_model.startprob_    = start_avg
    avg_model.transmat_     = trans_avg

    # 计算 avg_model 的 total logL
    total_ll_avg = 0.0
    idx = 0
    for L in lengths:
        seg = train_seq[idx:idx+L]
        total_ll_avg += avg_model.score(seg)
        idx += L

    print(f"[fit_pooled_hmm_fixed_emission_avg] "
          f"avg_model total logL = {total_ll_avg:.3f}, "
          f"best single logL = {ll_array.max():.3f}")

    return avg_model, ll_array


from scipy.special import logsumexp

def compute_state_probabilities(model, seq_01):
    """
    用 forward-backward 算每个 trial 对 3 个 state 的 posterior:
        gamma[t, k] = P(z_t = k | y_1...y_T)
    这里我们叫它 state probability，用来替换之前的 posterior 曲线。

    seq_01: shape (T,) 或 (T,1), 元素为 0/1
    返回: gamma, shape (T, n_states)
    """
    obs = np.asarray(seq_01, dtype=int).ravel()
    T_len = len(obs)
    K = model.n_components

    startprob = model.startprob_
    transmat  = model.transmat_
    emission  = model.emissionprob_

    log_start = np.log(startprob + 1e-15)
    log_trans = np.log(transmat + 1e-15)
    log_emit  = np.log(emission + 1e-15)

    # forward
    log_alpha = np.zeros((T_len, K))
    log_alpha[0] = log_start + log_emit[:, obs[0]]
    for t in range(1, T_len):
        for j in range(K):
            log_alpha[t, j] = logsumexp(log_alpha[t-1] + log_trans[:, j]) \
                              + log_emit[j, obs[t]]
    log_lik = logsumexp(log_alpha[-1])

    # backward
    log_beta = np.zeros((T_len, K))
    for t in range(T_len - 2, -1, -1):
        for i in range(K):
            log_beta[t, i] = logsumexp(
                log_trans[i, :] + log_emit[:, obs[t+1]] + log_beta[t+1, :]
            )

    log_gamma = log_alpha + log_beta - log_lik
    gamma = np.exp(log_gamma)
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma  # (T, K)

def load_pooled_3state_model_from_csv(csv_path):
    """
    从 export_model_params_to_csv 导出的 pooled 3-state HMM 参数 CSV 中
    重建一个 hmmlearn.CategoricalHMM 模型。

    期望的列格式（与你现在的 baseline CSV 一致）：
      - matrix      : 'startprob' / 'transmat' / 'emission'
      - from_state  : 状态 index（startprob/emission 用这个当 state）
      - to_state    : 对 transmat 来说是目标 state（其他行可以是 NaN）
      - obs         : 对 emission 来说是观测值 index 0/1（其他行 NaN）
      - value       : 参数数值
      其余列（state_name, to_state_name, obs_label）只用于可读性，可忽略。
    """

    df = pd.read_csv(csv_path)

    # ---- 推断状态数 & 观测数 ----
    # from_state 应该覆盖 0..(K-1)
    n_states = int(df["from_state"].max()) + 1

    # emission 行中的 obs 覆盖 0..(M-1)
    em_df = df[df["matrix"] == "emission"].copy()
    if not em_df.empty and em_df["obs"].notna().any():
        n_features = int(em_df["obs"].max()) + 1
    else:
        n_features = 2  # 默认 2（0/1）

    startprob = np.zeros(n_states, dtype=float)
    transmat  = np.zeros((n_states, n_states), dtype=float)
    emission  = np.zeros((n_states, n_features), dtype=float)

    # ---- startprob ----
    sp_df = df[df["matrix"] == "startprob"].copy()
    for _, row in sp_df.iterrows():
        s = int(row["from_state"])
        startprob[s] = float(row["value"])

    # ---- transmat ----
    tm_df = df[(df["matrix"] == "transmat") & df["to_state"].notna()].copy()
    for _, row in tm_df.iterrows():
        s_from = int(row["from_state"])
        s_to   = int(row["to_state"])
        transmat[s_from, s_to] = float(row["value"])

    # ---- emission ----
    em_df = df[(df["matrix"] == "emission") & df["obs"].notna()].copy()
    for _, row in em_df.iterrows():
        s   = int(row["from_state"])
        obs = int(row["obs"])
        emission[s, obs] = float(row["value"])

    # 归一化，避免数值累积误差
    if startprob.sum() > 0:
        startprob /= startprob.sum()

    # 每行转移概率归一化
    row_sums = transmat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transmat /= row_sums

    # 每个 state 的 emission 归一化
    row_sums = emission.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    emission /= row_sums

    # ---- 组装成 hmmlearn 模型 ----
    model = CategoricalHMM(
        n_components=n_states,
        n_iter=1,
        init_params=""  # 不让 hmmlearn 再改我们给定的参数
    )
    model.n_features    = n_features
    model.startprob_    = startprob
    model.transmat_     = transmat
    model.emissionprob_ = emission

    return model
