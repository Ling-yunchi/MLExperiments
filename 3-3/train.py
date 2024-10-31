import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    RandomHorizontalFlip,
    RandomRotation,
)
from tqdm import tqdm

from model import VitPetClassifier, ResnetPetClassifier

# 定义数据集路径和参数
batch_size = 32
image_size = 224
learning_rate_header = 1e-3  # 头部的学习率
epochs = 30
test_interval = 5

# 数据预处理
train_transform = Compose(
    [
        Resize((image_size, image_size)),
        RandomHorizontalFlip(),
        RandomRotation(10),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = Compose(
    [
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 加载数据集
dataset = datasets.OxfordIIITPet("data", split="trainval", transform=train_transform)
test_dataset = datasets.OxfordIIITPet("data", split="test", transform=test_transform)


# 划分数据集为训练和验证
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], torch.Generator().manual_seed(42)
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = ResnetPetClassifier(pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 冻结主干网络的权重
for param in model.resnet.parameters():
    param.requires_grad = False


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate_header,
)

work_dir = "runs/resnet/run_1"
writer = SummaryWriter(work_dir)

best_accuracy = 0.0

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(
        train_dataloader, desc=f"Training Epoch {epoch}/{epochs}", leave=False
    ):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_dataloader)
    writer.add_scalar("Loss/train", avg_loss, epoch)

    print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_loss}")

    # 验证阶段
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(
            val_dataloader, desc=f"Validation Epoch {epoch}/{epochs}", leave=False
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    writer.add_scalar("Accuracy/val", accuracy, epoch)
    print(f"Epoch {epoch}/{epochs}, Validation Accuracy: {accuracy}")

    torch.save(model.state_dict(), f"{work_dir}/epoch_{epoch}.pth")

    if (epoch) % test_interval == 0:
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(
                test_dataloader,
                desc=f"Test Epoch {epoch}/{epochs}",
                leave=False,
            ):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        writer.add_scalar("Accuracy/test", accuracy, epoch)
        print(f"Epoch {epoch}/{epochs}, Test Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f"{work_dir}/best_model.pth")
            print("Best test model saved.")

writer.close()
print("训练完成！")