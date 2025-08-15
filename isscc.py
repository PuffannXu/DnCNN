import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")
# 标题
ax.text(0.02, 1.07, "Challenge3: 算子适应力差异", fontsize=17, color='#d32f2f', weight='bold')
# ---------- 左侧：神经网络算子块组 ----------
left_ops = [
    {'name': 'Conv', 'color': '#a0c4ff', 'x':0.08, 'y':0.81},
    {'name': 'BatchNorm', 'color': '#bdb2ff', 'x':0.08, 'y':0.73},
    {'name': 'ReLU', 'color': '#ffd6a5', 'x':0.16, 'y':0.81},
    {'name': 'Pool', 'color': '#caff70', 'x':0.16, 'y':0.73},
]
ax.add_patch(FancyBboxPatch((0.05, 0.7), 0.19, 0.18, boxstyle="round,pad=0.012", lw=2, fc='#ececec', ec='#333'))
ax.text(0.145, 0.885, "典型神经网络算子\n(低层视觉)", fontsize=13, color='#173f63', ha='center', weight='bold')
# 画左侧的算子块
for op in left_ops:
    ax.add_patch(FancyBboxPatch((op['x'], op['y']), 0.07, 0.06, boxstyle="round,pad=0.012", lw=1.5, fc=op['color'], ec='#666'))
    ax.text(op['x']+0.035, op['y']+0.03, op['name'], ha='center', va='center', fontsize=12, weight='bold')
# 算子间连接线（简单水平线）
ax.plot([0.115, 0.195], [0.84, 0.84], color='#555', lw=2)
ax.plot([0.115, 0.195], [0.76, 0.76], color='#555', lw=2)
# ---------- 右侧：异构/特殊算子块组 ----------
right_ops = [
    {'name': 'Activation', 'color': '#d0f4de', 'x':0.73, 'y':0.78},
    {'name': 'FFT', 'color': '#fcd5ce', 'x':0.80, 'y':0.78},
    {'name': 'Sparsity', 'color': '#b2e7e8', 'x':0.73, 'y':0.70},
    {'name': 'Encode', 'color': '#f6c6ea', 'x':0.80, 'y':0.70},
]
ax.add_patch(FancyBboxPatch((0.70, 0.66), 0.20, 0.17, boxstyle="round,pad=0.012", lw=2, fc='#e3ecf0', ec='#333'))
ax.text(0.80, 0.845, "非神经网络/边缘算子", fontsize=13, color='#09527b', ha='center', weight='bold')
# 画右侧的算子块
for op in right_ops:
    ax.add_patch(FancyBboxPatch((op['x'], op['y']), 0.07, 0.06, boxstyle="round,pad=0.01", lw=1.5, fc=op['color'], ec='#666'))
    ax.text(op['x']+0.035, op['y']+0.03, op['name'], ha='center', va='center', fontsize=12, weight='bold')
# 算子间连接线
ax.plot([0.765, 0.835], [0.81, 0.81], color='#555', lw=2)
ax.plot([0.765, 0.835], [0.73, 0.73], color='#555', lw=2)
# ---------- 算子适配红色箭头与标签 ----------
ax.add_patch(FancyArrowPatch((0.24, 0.79), (0.70, 0.77), arrowstyle='-|>', color='#d32f2f', mutation_scale=32, lw=3))
ax.text(0.47, 0.8, "类型/资源/带宽适配难", fontsize=14, color='#d32f2f', ha='center', va='bottom')
# 算子类型差异波浪线
ax.plot([0.07, 0.9], [0.66, 0.66], color='#d32f2f', alpha=0.6, ls='--')
ax.text(0.48, 0.655, "算子类型、内存、带宽等资源差异", fontsize=12, color='#b14a00', ha='center')
# ---------- 底部碎片化资源区域与限制说明 ----------
ax.add_patch(FancyBboxPatch((0.21, 0.18), 0.58, 0.18, boxstyle="round,pad=0.015", lw=2, ec='#ba1b1b', fc='#faf5ee', alpha=0.99))
ax.text(0.5, 0.31, "资源碎片化 & 利用率低 & 扩展性受限", fontsize=14, color='#d32f2f', ha='center', weight='bold')
ax.text(0.5, 0.215, "算子异构导致资源利用碎片化，整体系统利用率受限，扩展性受制约", fontsize=11, color='#444', ha='center')
# 画些碎片小块状结构
for i in range(8):
    ax.add_patch(FancyBboxPatch((0.235+0.07*i, 0.22+0.008*(i%2)), 0.048, 0.03, boxstyle="round,pad=0.015", lw=1, ec='#ba1b1b', fc='#fff6f6', hatch='////'))
plt.tight_layout()
plt.show()