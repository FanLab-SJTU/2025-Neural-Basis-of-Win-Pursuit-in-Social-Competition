import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_rel

# Global settings
input_csvpath = "/Users/shuimuqinghua/Desktop/HMM/HMM PyProj Github/DEMO Male high rank.csv"
save_dir      = "/Users/shuimuqinghua/Desktop/HMM/HMM PyProj Github/Demo results/"
Exp_type      = "MH_naive"

os.makedirs(save_dir, exist_ok=True)
fig_dir   = os.path.join(save_dir, "Figures")
csv_dir   = os.path.join(save_dir, "Posterior_CSV")
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

BEST_K  = 3        # state number
N_INIT  = 256      # 每个 state 数的随机初始化次数（用于挑最优 fraction 3-state）
N_ITER  = 512      # HMM EM 迭代次数（每次 init 的上限）
# obs: 0 = go, 1 = win
# state0: Win-preferred   -> P(win)=0.95
# state1: Exploration     -> P(win)=0.50
# state2: Go-preferred    -> P(win)=0.05
EMISSION_FIXED = np.array([
    [0.05, 0.95],  # state 0: Win-like
    [0.50, 0.50],  # state 1: Random-like
    [0.95, 0.05],  # state 2: Go-like
], dtype=float)

# input all choice sequence
from HMM_Data_Helper import get_all_animals
all_animals = get_all_animals(input_csvpath)
print("Loaded animals and sequences:")
for aid, data in all_animals.items():
    print(f"Animal {aid}: train_len={len(data['train'])}, "
          f"day lengths={[len(data['days'][d]) for d in [0,1,2,3,4]]}")

# pool win days sequence
from HMM_Data_Helper import build_pooled_train
pooled_train_seq, pooled_lengths, used_animals = build_pooled_train(all_animals)
print(f"Pooled train trials: {len(pooled_train_seq)}, "
      f"used animals: {len(used_animals)}")

# train pooled 3-state CategoricalHMM
from HMM_modelFunc import fit_pooled_hmm_fixed_emission_avg
avg_model, all_lls = fit_pooled_hmm_fixed_emission_avg(
    pooled_train_seq,
    pooled_lengths,
    emission_fixed=EMISSION_FIXED,
    n_states=BEST_K,
    n_init=N_INIT,
    n_iter=N_ITER,
    random_seed=666666,
    top_frac=0.90
)

state_names = ["Win-like", "Random-like", "Go-like"]
print("Avg startprob:", avg_model.startprob_)
print("Avg transmat:\n", avg_model.transmat_)
print("Emission (fixed):\n", avg_model.emissionprob_)
best_model = avg_model

# export pooled model parameters
from HMM_Data_Helper import export_model_params_to_csv
param_csv_path = os.path.join(save_dir, f"{Exp_type}_pooled_3state_HMM_params.csv")
export_model_params_to_csv(best_model, state_names, param_csv_path)

# visualize pooled HMM transition matrix
plt.figure(figsize=(4, 3.5))
sns.heatmap(best_model.transmat_,
            annot=True, fmt=".2f",
            cmap="Blues", vmin=0, vmax=1,
            xticklabels=state_names,
            yticklabels=state_names)
plt.title(f"{Exp_type}: pooled 3-state HMM transition")
plt.xlabel("To state")
plt.ylabel("From state")
plt.tight_layout()
fig_tm_path = os.path.join(fig_dir, f"{Exp_type}_pooled_transition_matrix.pdf")
plt.savefig(fig_tm_path)
plt.show()
print(f"Transition matrix figure saved -> {fig_tm_path}")

# visualize choice and state change
from HMM_Plot_Helper import plot_behavior_and_posteriors
from HMM_modelFunc import compute_state_probabilities
summary_records = []

from HMM_modelFunc import load_pooled_3state_model_from_csv
baseline_param_path = "/Users/shuimuqinghua/Desktop/HMM/HMM PyProj Github/Baseline_3state_pooled_3state_HMM_params.csv"
baseline_model = load_pooled_3state_model_from_csv(baseline_param_path)
print("Loaded baseline 3-state model:")
print(" startprob:", baseline_model.startprob_)
print(" transmat:\n", baseline_model.transmat_)
print(" emission:\n", baseline_model.emissionprob_)

for aid, data in all_animals.items():
    day_seqs = data["days"]

    # 每只动物只初始化一次，收集 day 0–4 所有 trial 的结果
    rows_state = []

    # 5 行：day 0–4（包含 baseline）
    fig, axes = plt.subplots(5, 1, figsize=(9, 16), sharex=False)
    fig.suptitle(f"Animal {aid}: behavior & state probabilities",
                 fontsize=16)

    for idx, day in enumerate([0, 1, 2, 3, 4]):
        ax = axes[idx]
        seq = day_seqs.get(day, np.array([]).reshape(-1, 1))

        if seq.size == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.set_title(f"Day {day}")
            ax.set_axis_off()
            continue

        X = seq.reshape(-1, 1)
        # baseline 用 baseline_model，其余用 win-days pooled 的 best_model
        if day == 0:
            model_for_day = baseline_model
        else:
            model_for_day = best_model

        # 1. 用 pooled 的 best_model 计算 win days 的 state probability
        state_prob = compute_state_probabilities(model_for_day, seq)   # (T, 3)
        # state_prob_smooth = smooth_posteriors(state_prob, sigma=SMOOTH_SIGMA) # no smoothing
        state_prob_smooth = state_prob

        # 2. 可视化（baseline 也会画出来）
        plot_behavior_and_posteriors(
            ax, seq, state_prob_smooth, day, state_names
        )

        # 3. 统计：均值 state probability + dominant fraction
        mean_state = state_prob_smooth.mean(axis=0)   # (3,)
        dom_idx    = np.argmax(state_prob_smooth, axis=1)
        dom_frac   = np.array([
            np.mean(dom_idx == 0),
            np.mean(dom_idx == 1),
            np.mean(dom_idx == 2)
        ])

        summary_records.append(dict(
            animal_id=aid,
            day=day,
            n_trials=len(seq),
            mean_state_win=mean_state[0],
            mean_state_random=mean_state[1],
            mean_state_go=mean_state[2],
            dom_frac_win=dom_frac[0],
            dom_frac_random=dom_frac[1],
            dom_frac_go=dom_frac[2]
        ))

        # 4. 保存 trial-level state prob + choice（0–4 天都 append 到同一个 rows_state 里）
        for t in range(len(seq)):
            rows_state.append(dict(
                animal_id=aid,
                day=day,
                trial_in_day=t + 1,
                choice=int(seq[t, 0]),
                state_win=state_prob_smooth[t, 0],
                state_random=state_prob_smooth[t, 1],
                state_go=state_prob_smooth[t, 2]
            ))

    # 图例放第一行
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = os.path.join(fig_dir, f"{aid}_state_prob_all_days.pdf")
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"[Saved] figure for {aid} -> {fig_path}")

    # per-animal state prob CSV：包含 day 0–4 所有 trial
    df_state = pd.DataFrame(rows_state)
    csv_path_animal = os.path.join(csv_dir, f"{aid}_state_prob_and_choice.csv")
    df_state.to_csv(csv_path_animal, index=False)
    print(f"[Saved] state prob CSV for {aid} -> {csv_path_animal}")

# 全动物 × 全天 summary
summary_df = pd.DataFrame(summary_records)
summary_csv_path = os.path.join(save_dir, f"{Exp_type}_state_posterior_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"Summary of mean state prob & dominant fraction saved -> {summary_csv_path}")
