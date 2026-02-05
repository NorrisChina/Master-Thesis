import numpy as np
import matplotlib.pyplot as plt

# === 参数设置 ===
radii = [2, 10, 30]  # R=2 (精密), R=10 (中间), R=30 (Safe Harbor)
deltas = np.logspace(0, -2, 100) # 更密一点，画图更平滑
t_hash_us = 2.13

# === 绘图 ===

# 统一所有字体为较小一致
UNIFORM_SIZE = 20
fig, ax = plt.subplots(figsize=(7, 5))  # 推荐：宽7英寸，高5英寸
UNIFORM_SIZE = 12
plt.rc('font', size=UNIFORM_SIZE)
plt.rc('axes', titlesize=UNIFORM_SIZE)
plt.rc('axes', labelsize=UNIFORM_SIZE)
plt.rc('xtick', labelsize=UNIFORM_SIZE)
plt.rc('ytick', labelsize=UNIFORM_SIZE)
plt.rc('legend', fontsize=UNIFORM_SIZE)
plt.rc('figure', titlesize=UNIFORM_SIZE)

# 颜色映射
colors = ['tab:green', 'tab:orange', 'tab:red']

for i, R in enumerate(radii):
    # 计算 Cost
    set_sizes = np.ceil((2 * R) / deltas) + 1
    latency_ms = (set_sizes * t_hash_us) / 1000.0 
    # 画曲线
    label_text = f'$R={R}$'
    ax.plot(deltas, latency_ms, lw=2.5, color=colors[i], label=label_text)

    # 1. 标注极端情况 (Delta=0.01) - 展示瓶颈
    cost_fine = latency_ms[-1] # 最后一个点是 0.01
    ax.text(0.01, cost_fine, f'{cost_fine:.1f}', 
            fontsize=UNIFORM_SIZE, color=colors[i], fontweight='bold', ha='left', va='center')

    # 2. 标注你的选择 (Delta=0.1) - 展示合理性
    # 找到 Delta=0.1 对应的延迟
    idx_01 = np.abs(deltas - 0.1).argmin()
    cost_01 = latency_ms[idx_01]
    # 画个圆点
    ax.plot(0.1, cost_01, 'o', color=colors[i], markersize=8, markeredgecolor='white')
    # 添加文字标注 (错开一点位置防止重叠)
    offset_y = -0.5 if R==2 else 0.2
    ax.text(0.1, cost_01 + offset_y, f'{cost_01:.1f}', 
            fontsize=UNIFORM_SIZE, color=colors[i], fontweight='bold', ha='center')
# === 装饰图表 ===
# 绘制垂直线标记 Selected Operating Point
ax.axvline(0.1, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
ax.text(0.1, ax.get_ylim()[1]*0.85, 'Selected\nOperating Point\n($\Delta=0.1$)', 
        color='gray', ha='center', fontsize=UNIFORM_SIZE, backgroundcolor='white')

# 绘制水平线标记 Soft Real-time Limit
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.text(0.5, 1.1, 'Soft Real-time Limit (~1ms)', color='gray', fontsize=UNIFORM_SIZE)

ax.set_xscale('log')
ax.invert_xaxis() # 从 1.0 -> 0.01




ax.set_xlabel('Quantization Resolution $\Delta$ (Log Scale)', fontsize=UNIFORM_SIZE)
ax.set_ylabel('Decoder Compute Latency (ms)', fontsize=UNIFORM_SIZE)
ax.set_title('Computational Scalability: The Cost of Safety', fontsize=UNIFORM_SIZE)
ax.tick_params(axis='both', which='major', labelsize=UNIFORM_SIZE)
ax.tick_params(axis='both', which='minor', labelsize=UNIFORM_SIZE)
ax.legend(loc='upper left', fontsize=UNIFORM_SIZE, frameon=True)
ax.grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()
plt.savefig('decoder_cost_multiline_annotated.png', dpi=300)
plt.show()