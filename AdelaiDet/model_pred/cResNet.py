import torch
import torch.nn as nn
import torchvision.models as models


class ResNetModel(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNetModel, self).__init__()
        # 使用预训练的 ResNet50
        self.resnet = models.resnet50(pretrained=True)

        # 去掉 ResNet 的最后一层全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # 特征融合后的全连接层
        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 256)
        self.classifier = nn.Linear(2048 + 256, num_classes)

    def forward(self, x, p):
        # 提取 ResNet 特征
        features = self.resnet(x).view(x.size(0), -1)

        # 对输入 p 进行两次全连接升维
        p = self.fc1(p)  # 20 -> 128
        p = torch.relu(p)
        p = self.fc2(p)  # 128 -> 128
        p = torch.relu(p)

        # 特征融合
        combined_features = torch.cat((features, p), dim=1)

        # 分类
        output = self.classifier(combined_features)
        return output