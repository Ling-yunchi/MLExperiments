import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
train_loss_df = pd.read_csv("rundata/train_loss.csv")
train_acc_df = pd.read_csv("rundata/train_acc.csv")
test_acc_df = pd.read_csv("rundata/test_acc.csv")

# 提取数据
train_loss = train_loss_df["Value"].values
train_acc = train_acc_df["Value"].values
train_epochs = train_loss_df["Step"].values

# 测试数据的步长为10
test_acc = test_acc_df["Value"].values
test_epochs = test_acc_df["Step"].values

# 创建训练集图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# 训练集损失
color = "tab:blue"
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss", color=color)
(line1,) = ax1.plot(train_epochs, train_loss, label="Training Loss", color=color)
ax1.tick_params(axis="y", labelcolor=color)

# 训练集准确率
ax1b = ax1.twinx()
color = "tab:green"
ax1b.set_ylabel("Accuracy", color=color)
(line2,) = ax1b.plot(train_epochs, train_acc, label="Training Accuracy", color=color)
ax1b.tick_params(axis="y", labelcolor=color)

# 添加图例
lines = [line1, line2]
ax1.legend(lines, [l.get_label() for l in lines], loc="center right")

# 添加标题
ax1.set_title("Training Loss and Accuracy Over Epochs")

# 测试集准确率
color = "tab:red"
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy", color=color)
ax2.plot(test_epochs, test_acc, label="Test Accuracy", color=color, marker="o")
ax2.tick_params(axis="y", labelcolor=color)

# 添加图例
ax2.legend(loc="upper left")

# 添加标题
ax2.set_title("Test Accuracy Over Epochs")

# 调整布局
fig.tight_layout()

# 保存和显示图形
plt.savefig("separate_train_test_plots.png")
plt.show()
