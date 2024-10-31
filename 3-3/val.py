import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
from model import VitPetClassifier

# 定义数据集路径和参数
batch_size = 64
image_size = 224

# 数据预处理
transform = Compose(
    [
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 加载测试数据集
test_dataset = datasets.OxfordIIITPet("data", split="test", transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = VitPetClassifier()
model.load_state_dict(torch.load("runs/vit/run_1/best_model.pth"))

# 加载最佳模型权重
# model.load_state_dict(torch.load("best_model.pth"))

# 将模型转移到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 设置模型为评估模式

# 评估循环
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_dataloader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # 获取预测的类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # 计算正确的预测数量

# 计算准确率
accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")
