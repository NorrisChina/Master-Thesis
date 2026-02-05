import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

# === 1. 参数设置 (The Genius Configuration) ===
mu = 50.0
sigma_gaussian = 16.0  # 标准差 sigma = 16 (即方差 s^2 = 256)
# 解释: 3*sigma = 48, 范围 [2, 98], 完美适配 [0, 100]

# 计算方差匹配的 Uniform 半宽
# Var(U) = a^2 / 3 = sigma^2  => a = sigma * sqrt(3)
a_uniform = sigma_gaussian * np.sqrt(3)  # approx 27.71

# 物理边界
x_min, x_max = 0, 100
x_range = np.linspace(x_min, x_max, 2000)

# === 2. 计算 PDF ===
# Gaussian
pdf_gaussian = norm.pdf(x_range, loc=mu, scale=sigma_gaussian)

# Uniform
# scipy 的 uniform 参数: loc=起点, scale=宽度(2a)
uniform_lower = mu - a_uniform
uniform_width = 2 * a_uniform
pdf_uniform = uniform.pdf(x_range, loc=uniform_lower, scale=uniform_width)

# === 3. 开始绘图 (左右布局) ===
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

# --- 左图: Gaussian ---
ax0 = axes[0]
ax0.plot(x_range, pdf_gaussian, 'b-', lw=2.5, label=f'$\sigma={int(sigma_gaussian)}$')
ax0.fill_between(x_range, pdf_gaussian, color='blue', alpha=0.15)

# 标注物理边界 (0 和 100)
ax0.axvline(x_min, color='k', lw=3, linestyle='-', alpha=0.3)
ax0.axvline(x_max, color='k', lw=3, linestyle='-', alpha=0.3)
ax0.text(x_min+1, max(pdf_gaussian)*0.9, 'Wall (0)', rotation=90, verticalalignment='top', alpha=0.5)
ax0.text(x_max-3, max(pdf_gaussian)*0.9, 'Wall (100)', rotation=90, verticalalignment='top', alpha=0.5)

# 标注 3-sigma 范围
sigma3_lower = mu - 3 * sigma_gaussian  # 50 - 48 = 2
sigma3_upper = mu + 3 * sigma_gaussian  # 50 + 48 = 98
ax0.axvline(sigma3_lower, color='r', linestyle=':', alpha=0.8, lw=1.5)
ax0.axvline(sigma3_upper, color='r', linestyle=':', alpha=0.8, lw=1.5)

# --- Set font sizes for all elements ---
LARGE = 20
MEDIUM = 20
SMALL = 20

ax0.text(sigma3_lower + 2, max(pdf_gaussian)*0.55, r'$-3\sigma$ (2.0)', color='r', ha='left', fontsize=SMALL, fontweight='bold')
ax0.text(sigma3_upper - 2, max(pdf_gaussian)*0.55, r'$+3\sigma$ (98.0)', color='r', ha='right', fontsize=SMALL, fontweight='bold')

ax0.set_title(f'(a) Baseline Gaussian Model', fontsize=LARGE)
ax0.set_xlabel('Robot State $S_t$', fontsize=MEDIUM)
ax0.set_ylabel('Probability Density', fontsize=MEDIUM)
ax0.legend(loc='upper right', fontsize=MEDIUM, frameon=True)
ax0.grid(True, alpha=0.3)
ax0.set_xlim(x_min, x_max)
ax0.tick_params(axis='both', which='major', labelsize=MEDIUM)

# --- 右图: Uniform ---
ax1 = axes[1]
ax1.plot(x_range, pdf_uniform, 'g-', lw=2.5, label=f'$a={a_uniform:.1f}$')
ax1.fill_between(x_range, pdf_uniform, color='green', alpha=0.15)

# 标注物理边界
ax1.axvline(x_min, color='k', lw=3, linestyle='-', alpha=0.3)
ax1.axvline(x_max, color='k', lw=3, linestyle='-', alpha=0.3)


# 标注 Uniform 边界
ax1.axvline(uniform_lower, color='g', linestyle='--', lw=2)
ax1.axvline(uniform_lower + uniform_width, color='g', linestyle='--', lw=2)
# 显示左右边界的横坐标
ax1.text(uniform_lower, max(pdf_uniform)*0.05, f'{uniform_lower:.1f}', color='g', ha='left', va='bottom', fontsize=SMALL, fontweight='bold')
ax1.text(uniform_lower + uniform_width, max(pdf_uniform)*0.05, f'{(uniform_lower + uniform_width):.1f}', color='g', ha='right', va='bottom', fontsize=SMALL, fontweight='bold')

ax1.set_title(f'(b) Variance-Matched Uniform Model', fontsize=LARGE)
ax1.set_xlabel('Robot State $S_t$', fontsize=MEDIUM)
ax1.legend(loc='upper right', fontsize=MEDIUM, frameon=True)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(x_min, x_max)
ax1.tick_params(axis='both', which='major', labelsize=MEDIUM)


# === 总标题 ===
plt.suptitle(f'Error Model Comparison: Optimal Space Utilization ($\sigma={int(sigma_gaussian)}$, $\sigma^2={int(sigma_gaussian**2)}$)', 
             fontsize=LARGE, y=1.05)

plt.tight_layout()
# 保存图片
plt.savefig('error_models_sigma16.png', dpi=300, bbox_inches='tight')
plt.show()