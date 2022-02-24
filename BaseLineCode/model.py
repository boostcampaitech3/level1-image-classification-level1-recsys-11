import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

# 
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

# ViT
import torch
from vit_pytorch import ViT

class Vit(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.v = ViT(
            image_size = 128, # Image size. If you have rectangular images, make sure your image size is the maximum of the width and height
            patch_size = 32,
            num_classes = num_classes, # Number of classes to classify.
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
            )   

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.v(x)
        return x


from efficientnet_pytorch import EfficientNet
import math

# EfficientNet
class EfficientNetFineTuning(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7')
        self.efficientnet._fc = nn.Linear(in_features=2560, out_features = num_classes) 
        # self.linear = nn.Linear(in_features = self.efficientnet._fc.out_features, out_features = num_classes)

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.efficientnet._fc.weight)
        stdv = 1/math.sqrt(self.efficientnet._fc.in_features)
        self.efficientnet._fc.bias.data.uniform_(-stdv, stdv)

        # initialize w & b
        # torch.nn.init.xavier_uniform_(self.linear.weight)
        # stdv = 1/math.sqrt(self.linear.in_features)
        # self.linear.bias.data.uniform_(-stdv, stdv)
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.efficientnet(x)
        # x = self.linear(x)
        return x


class EfficientNetB0FeatureExtraction(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        # FREEZE the paramters
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        self.efficientnet._fc.requires_grad_ = True

        # trainable layer added and convert last layer trainable
        self.efficientnet._fc = nn.Linear(in_features=1280, out_features = 450) # trainable
        self.linear_1 = nn.Linear(in_features = 450, out_features = 90) # trainable
        self.linear_2 = nn.Linear(in_features = 90, out_features = num_classes) # trainable

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.efficientnet._fc.weight)
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)

        stdv = 1/math.sqrt(self.efficientnet._fc.in_features)
        self.efficientnet._fc.bias.data.uniform_(-stdv, stdv)
        stdv = 1/math.sqrt(self.linear_1.in_features)
        self.linear_1.bias.data.uniform_(-stdv, stdv)
        stdv = 1/math.sqrt(self.linear_2.in_features)
        self.linear_2.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x


class ResNet18F(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(in_features=512, out_features = num_classes, bias= True) 
        # self.Linear_1 = nn.Linear(in_features=1000, out_features = num_classes)

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.resnet18.fc.weight)
        stdv = 1/math.sqrt(self.resnet18.fc.in_features)
        self.resnet18.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.resnet18(x)
        return x