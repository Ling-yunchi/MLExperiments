from torch import nn
from torchvision import models
from transformers import ViTModel


class ResnetPetClassifier(nn.Module):
    def __init__(self, num_classes=37, pretrained=True):
        super(ResnetPetClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Identity()
        self.linear = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids):
        output = self.resnet(input_ids)  # b 2048
        output = self.linear(output)
        return output


class VitPetClassifier(nn.Module):
    def __init__(self, num_classes=37, pretrained=True):
        super(VitPetClassifier, self).__init__()
        self.vit = (
            ViTModel.from_pretrained("google/vit-base-patch16-224")
            if pretrained
            else ViTModel()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Sequential(
            nn.Linear(768, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        vit_output = self.vit(x).last_hidden_state.permute(0, 2, 1)  # b 768 197
        output = self.pool(vit_output).squeeze(-1)
        output = self.output(output)
        return output
