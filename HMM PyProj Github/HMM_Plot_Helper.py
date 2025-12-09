import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def alpha_map(x, k=16):
    """把 [0,1] 概率映射到可视化 alpha，用于 dominant state shading。"""
    s = 1 / (1 + np.exp(-k * (x - 0.5)))
    s0 = 1 / (1 + np.exp(k * 0.5))
    s1 = 1 / (1 + np.exp(-k * 0.5))
    return (s - s0) / (s1 - s0)

def plot_behavior_and_posteriors_pp(ax, seq, post_smooth, day, state_names):
    """
    单个 day 的轨迹：行为选择、三条 posterior 曲线 + dominant state 着色。
    seq: 0/1
    post_smooth: (T, 3), state 顺序已经是 [Win-like, Random-like, Go-like]
    """
    T = len(seq)
    time = np.arange(1, T + 1)

    # 行为 scatter
    seq = np.asarray(seq).ravel()
    ax.scatter(time[seq == 1], np.ones_like(seq[seq == 1]),
               color='black', marker='o', s=25, label='Win (1)', zorder=5)
    ax.scatter(time[seq == 0], np.zeros_like(seq[seq == 0]),
               marker='o', facecolors='none', edgecolors='black',
               s=25, label='Go (0)', zorder=5)

    # posterior 曲线
    win_probs    = post_smooth[:, 0]
    random_probs = post_smooth[:, 1]
    go_probs     = post_smooth[:, 2]

    ax.plot(time, win_probs,    color='red',   linewidth=2, label=state_names[0])
    ax.plot(time, random_probs, color=[1, 229/255, 41/255], linewidth=2, label=state_names[1])
    ax.plot(time, go_probs,     color='blue',  linewidth=2, label=state_names[2])
    
    # # posterior 曲线 2states
    # win_probs    = post_smooth[:, 0]
    # # random_probs = post_smooth[:, 1]
    # go_probs     = post_smooth[:, 1]

    # ax.plot(time, win_probs,    color='red',   linewidth=2, label=state_names[0])
    # # ax.plot(time, random_probs, color=[1, 229/255, 41/255], linewidth=2, label=state_names[1])
    # ax.plot(time, go_probs,     color='blue',  linewidth=2, label=state_names[1])

    # dominant state 着色
    dom_idx = np.argmax(post_smooth, axis=1)   # 0/1/2
    state_colors = ['mistyrose', 'lightyellow', 'lightblue']
    # state_colors = ['mistyrose', 'lightblue']

    for t in range(T):
        s_idx = dom_idx[t]
        alpha = alpha_map(post_smooth[t, s_idx])
        ax.axvspan(t + 0.5, t + 1.5, color=state_colors[s_idx],
                   alpha=alpha, linewidth=0)

    ax.set_xlim(0.5, T + 0.5)
    ax.set_ylim(-0.2, 1.05)
    ax.set_ylabel("Probability")
    ax.set_title(f"Day {day}")
    ax.grid(True, alpha=0.3)
    
def plot_behavior_and_posteriors(ax, seq, state_prob_smooth, day, state_names):
    T = len(seq)
    time = np.arange(1, T + 1)

    seq = np.asarray(seq).ravel()
    ax.scatter(time[seq == 1], np.ones_like(seq[seq == 1]),
               color='black', marker='o', s=25, label='Win (1)', zorder=5)
    ax.scatter(time[seq == 0], np.zeros_like(seq[seq == 0]),
               marker='o', facecolors='none', edgecolors='black',
               s=25, label='Go (0)', zorder=5)

    win_probs    = state_prob_smooth[:, 0]
    random_probs = state_prob_smooth[:, 1]
    go_probs     = state_prob_smooth[:, 2]

    ax.plot(time, win_probs,    color='red',   linewidth=2, label=state_names[0])
    ax.plot(time, random_probs, color=[1, 229/255, 41/255], linewidth=2, label=state_names[1])
    ax.plot(time, go_probs,     color='blue',  linewidth=2, label=state_names[2])

    dom_idx = np.argmax(state_prob_smooth, axis=1)
    state_colors = ['mistyrose', 'lightyellow', 'lightblue']
    for t in range(T):
        s_idx = dom_idx[t]
        alpha = alpha_map(state_prob_smooth[t, s_idx])
        ax.axvspan(t + 0.5, t + 1.5, color=state_colors[s_idx],
                   alpha=alpha, linewidth=0)

    ax.set_xlim(0.5, T + 0.5)
    ax.set_ylim(-0.2, 1.05)
    ax.set_ylabel("State probability")
    ax.set_title(f"Day {day}")
    ax.grid(True, alpha=0.3)
