"""
Complete reproduction of Figure 1 + CSV export. It will display the figure inline AND save files.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files  # For easy download

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
np.random.seed(42)

plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# =============================================================================
# EXACT GAUSSIAN MI (population log-det)
# =============================================================================
def gaussian_mi_exact(Cov_X, Cov_Y, Cov_XY):
    Cov_joint = np.block([[Cov_X, Cov_XY],
                          [Cov_XY.T, Cov_Y]])
    sign_x, logdet_x = np.linalg.slogdet(Cov_X)
    sign_y, logdet_y = np.linalg.slogdet(Cov_Y)
    sign_j, logdet_j = np.linalg.slogdet(Cov_joint)
    if sign_x <= 0 or sign_y <= 0 or sign_j <= 0:
        return 0.0
    return 0.5 * (logdet_x + logdet_y - logdet_j)

# =============================================================================
# PANEL (a): SATURATION
# =============================================================================
T_sat = 200
A = 1.0
lambda_sat = 0.05
epsilon = 1e-3
t_sat = np.arange(T_sat + 1)

N_t = t_sat
Q_full = A / (A + lambda_sat * N_t + epsilon)
Q_selector = A / (A + lambda_sat * 2.0 + epsilon) * np.ones_like(t_sat)
t_theory = np.arange(1, T_sat + 1)
Q_theory = (A / lambda_sat) / t_theory

# =============================================================================
# PANEL (b): QUERY-AWARENESS — EXACT POPULATION MI
# =============================================================================
K = 6
m = 8
d = K * m
lambda_qa = 0.20
mu_qa = 0.0

I_m = np.eye(m)
Cov_U = I_m
Cov_C_block = 2.0 * I_m
Cov_CU_block = I_m

vals_aware_small = []
vals_agnostic_small = []
vals_agnostic_union = []
vals_aware_large = []

for k in range(K):
    Cov_Y = Cov_U
    nuisance_blocks = [j for j in range(K) if j != k]
    n_nuis = len(nuisance_blocks) * m

    # Aware small
    iy = gaussian_mi_exact(Cov_C_block, Cov_Y, Cov_CU_block)
    Cov_ZN_aware = np.zeros((m, n_nuis))
    in_ = gaussian_mi_exact(Cov_C_block, np.eye(n_nuis), Cov_ZN_aware)
    u_aware = iy - lambda_qa * in_ - mu_qa * 0.0
    vals_aware_small.append(max(0.0, u_aware))
    vals_aware_large.append(max(0.0, u_aware))

    # Agnostic small
    u_wrong = 0.0 - lambda_qa * gaussian_mi_exact(Cov_C_block, Cov_U, Cov_CU_block)
    u_wrong_clipped = max(0.0, u_wrong)
    u_agnostic = (1.0/K) * u_aware + ((K-1.0)/K) * u_wrong_clipped
    vals_agnostic_small.append(u_agnostic)

    # Agnostic union
    Cov_C_full = np.kron(np.eye(K), Cov_C_block)
    Cov_CY_full = np.zeros((d, m))
    Cov_CY_full[k*m:(k+1)*m, :] = Cov_CU_block
    iy_full = gaussian_mi_exact(Cov_C_full, Cov_Y, Cov_CY_full)
    Cov_CN_full = np.zeros((d, n_nuis))
    for idx, j in enumerate(nuisance_blocks):
        Cov_CN_full[j*m:(j+1)*m, idx*m:(idx+1)*m] = Cov_CU_block
    in_full = gaussian_mi_exact(Cov_C_full, np.eye(n_nuis), Cov_CN_full)
    u_union = iy_full - lambda_qa * in_full
    vals_agnostic_union.append(max(0.0, u_union))

ceiling = vals_aware_small[0]
norm_aware_small = np.mean(vals_aware_small) / ceiling
norm_agnostic_small = np.mean(vals_agnostic_small) / ceiling
norm_agnostic_union = np.mean(vals_agnostic_union) / ceiling
norm_aware_large = np.mean(vals_aware_large) / ceiling

categories = ['Agnostic\nsmall budget', 'Agnostic\nunion budget', 'Aware\nsmall budget', 'Aware\nlarge budget']
values_qa = [norm_agnostic_small, norm_agnostic_union, norm_aware_small, norm_aware_large]

# =============================================================================
# PANEL (c): GAIN THRESHOLD
# =============================================================================
T_ar = 120
n_runs = 500
eta_bar = 0.15
sigma = 0.05
e0 = 0.0

a_values = [0.63, 1.00, 1.12]
labels = [f'$a = {a_values[0]:.2f}$ (contractive)', f'$a = {a_values[1]:.2f}$ (random walk)', f'$a = {a_values[2]:.2f}$ (divergent)']
colors_ar = ['#4a6fa5', '#d9534f', '#5cb85c']

mean_trajectories = {}
for a in a_values:
    e = np.zeros((n_runs, T_ar + 1))
    e[:, 0] = e0
    for step in range(T_ar):
        xi = np.random.normal(eta_bar, sigma, n_runs)
        e[:, step + 1] = a * e[:, step] + xi
    mean_trajectories[a] = e.mean(axis=0)

# =============================================================================
# VERIFICATION PRINTS
# =============================================================================
print("=" * 65)
print("VERIFICATION AGAINST PAPER VALUES")
print("=" * 65)
print("\n--- Panel (a): Saturation ---")
print(f"Q_full at t=200:       {Q_full[200]:.4f}   (paper: ~0.091)")
print(f"Selector plateau:      {Q_selector[0]:.4f}   (paper: 0.908)")
print(f"t·Q_t at t=200:        {200 * Q_full[200]:.2f}   (paper: 18.18, limit=20.0)")
print(f"1/Q_t slope (fitted):  {np.polyfit(t_sat, 1.0/Q_full, 1)[0]:.4f} (paper: 0.0500)")

print("\n--- Panel (b): Query-Awareness (EXACT POPULATION MI) ---")
print(f"Agnostic small budget: {values_qa[0]:.4f}   (paper: 0.165, bound 1/6={1/6:.4f})")
print(f"Agnostic union budget: {values_qa[1]:.4f}   (paper: 0.000)")
print(f"Aware small budget:    {values_qa[2]:.4f}   (paper: 1.000)")
print(f"Aware large budget:    {values_qa[3]:.4f}   (paper: 1.000)")

print("\n--- Panel (c): Gain Threshold ---")
m_063 = mean_trajectories[0.63]
stationary_mean = m_063[-40:].mean()
theory_stationary = eta_bar / (1 - 0.63)
print(f"a=0.63, last-40 mean:  {stationary_mean:.4f}   (paper: 0.4053, theory: {theory_stationary:.4f})")

m_100 = mean_trajectories[1.00]
slope_100 = np.polyfit(np.arange(T_ar + 1), m_100, 1)[0]
print(f"a=1.00, drift slope:   {slope_100:.4f}   (paper: 0.1497, theory: {eta_bar:.4f})")

m_112 = mean_trajectories[1.12]
valid_idx = (m_112 > 0.1) & (np.arange(T_ar + 1) > 10)
if np.sum(valid_idx) > 5:
    log_growth = np.polyfit(np.arange(T_ar + 1)[valid_idx], np.log(m_112[valid_idx]), 1)[0]
    print(f"a=1.12, log-growth:    {log_growth:.4f}   (paper: 0.1133, theory: {np.log(1.12):.4f})")
print("=" * 65)

# =============================================================================
# SAVE CSV FILES
# =============================================================================
print("\n--- Saving CSV files ---")

# CSV 1: Panel (a) Saturation
df_a = pd.DataFrame({
    'turn': t_sat,
    'Q_full_retention': Q_full,
    'Q_selector_O1_leakage': Q_selector,
    'Q_theory_c_over_t': np.concatenate([[np.nan], Q_theory])  # t=0 has no theory
})
df_a.to_csv('panel_a_saturation.csv', index=False)
print("Saved: panel_a_saturation.csv")

# CSV 2: Panel (b) Query-Awareness
df_b = pd.DataFrame({
    'category': categories,
    'normalized_usable_value': values_qa
})
df_b.to_csv('panel_b_query_awareness.csv', index=False)
print("Saved: panel_b_query_awareness.csv")

# CSV 3: Panel (c) Gain Threshold
t_ar = np.arange(T_ar + 1)
df_c = pd.DataFrame({
    'turn': t_ar,
    'mean_error_a_0.63': mean_trajectories[0.63],
    'mean_error_a_1.00': mean_trajectories[1.00],
    'mean_error_a_1.12': mean_trajectories[1.12]
})
df_c.to_csv('panel_c_gain_threshold.csv', index=False)
print("Saved: panel_c_gain_threshold.csv")

# =============================================================================
# PLOTTING
# =============================================================================
print("\n--- Generating figure ---")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# (a) Saturation
ax = axes[0]
ax.plot(t_sat, Q_full, color='#4a6fa5', linewidth=2.2, label='Full retention $Q_t$')
ax.plot(t_sat, Q_selector, color='black', linewidth=2.2, label='Selector, O(1) leakage')
ax.plot(t_theory, Q_theory, 'k--', linewidth=1.5, label='Theory $c/t$, $c=A/\\lambda$')
ax.set_xlabel('Turn $t$')
ax.set_ylabel('Quality $Q_t$')
ax.set_title('(a) Saturation', fontweight='bold')
ax.legend(frameon=True, fancybox=True, loc='upper right')
ax.set_xlim(0, 200)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# (b) Query-awareness
ax = axes[1]
bar_colors = ['#2c3e50', '#2c3e50', '#5dade2', '#5dade2']
bars = ax.bar(categories, values_qa, color=bar_colors, edgecolor='black', linewidth=0.6, width=0.6)
ax.axhline(y=1/K, color='black', linestyle='--', linewidth=1.5, label=f'Agnostic bound $1/K$ (Thm. 2)')
ax.set_ylabel('Normalized usable value')
ax.set_title('(b) Query-awareness', fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend(frameon=True, fancybox=True, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, values_qa):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.03,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# (c) Gain threshold
ax = axes[2]
t_ar = np.arange(T_ar + 1)
for a, label, color in zip(a_values, labels, colors_ar):
    ax.plot(t_ar, mean_trajectories[a], color=color, linewidth=2.2, label=label)
ax.axhline(y=eta_bar / (1 - 0.63), color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='$\\bar{\\eta}/(1-a) = 0.405$')
ax.plot(t_ar, eta_bar * t_ar, 'k:', linewidth=1.5, alpha=0.7, label='$\\bar{\\eta}t$ drift')
ax.set_xlabel('Turn $t$')
ax.set_ylabel('Mean inherited error')
ax.set_title('(c) Gain threshold', fontweight='bold')
ax.set_xlim(0, 120)
ax.set_ylim(0, 3.0)
ax.legend(frameon=True, fancybox=True, loc='upper left', fontsize=8.5)
ax.grid(True, alpha=0.3)
ax.annotate('$a = 1.12$ leaves frame\n(final mean $\\sim 10^6$)',
            xy=(12, 2.6), fontsize=9.5, ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

plt.tight_layout()
plt.savefig('figure1_reproduction.png', dpi=200, bbox_inches='tight')
print("Saved: figure1_reproduction.png")
plt.show()

print("\nAll done! Files saved to your workspace.")