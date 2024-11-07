import matplotlib.pyplot as plt

# 数据
vocab_sizes = [1000, 5000, 10000, 20000, 30000]
accuracies = [0.8593, 0.9339, 0.9496, 0.9521, 0.9500]
times = [0.6098, 1.0191, 1.5702, 2.4185, 3.7728]

# 创建图形
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制准确率曲线
color = "tab:blue"
ax1.set_xlabel("Vocabulary Size")
ax1.set_ylabel("Accuracy", color=color)
ax1.plot(vocab_sizes, accuracies, color=color, marker="o", label="Accuracy")
ax1.tick_params(axis="y", labelcolor=color)

# 创建第二个 y 轴
ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Time (seconds)", color=color)
ax2.plot(vocab_sizes, times, color=color, marker="x", linestyle="--", label="Time")
ax2.tick_params(axis="y", labelcolor=color)

# 添加图例
# fig.tight_layout()
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper left")

# 添加标题
plt.title("Vocabulary Size vs. Accuracy and Training Time")

# 显示图形
plt.show()
