import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import math

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


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


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # self.Linear_1 = nn.Linear(in_features=1000, out_features = num_classes)

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.resnet18.fc.weight)
        stdv = 1 / math.sqrt(self.resnet18.fc.in_features)
        self.resnet18.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.resnet18(x)
        return x


class ResNet18Freeze(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)

        for param in self.resnet18.parameters():
            param.requires_grad = False

        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        # self.Linear_1 = nn.Linear(in_features=1000, out_features = num_classes)

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.resnet18.fc.weight)
        stdv = 1 / math.sqrt(self.resnet18.fc.in_features)
        self.resnet18.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.resnet18(x)
        return x


# EfficientNet



class EfficientNetB3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b3')

        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(in_features=in_features, out_features=num_classes)
        # self.linear = nn.Linear(in_features = self.efficientnet._fc.out_features, out_features = num_classes)

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.efficientnet._fc.weight)
        stdv = 1 / math.sqrt(self.efficientnet._fc.in_features)
        self.efficientnet._fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.efficientnet(x)
        return x


class EfficientNetB3Freeze(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b3')

        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # Drop out will be added Here
        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(in_features=in_features, out_features=num_classes)
        # self.linear = nn.Linear(in_features = self.efficientnet._fc.out_features, out_features = num_classes)

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.efficientnet._fc.weight)
        stdv = 1 / math.sqrt(self.efficientnet._fc.in_features)
        self.efficientnet._fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.efficientnet(x)
        return x
    
class EfficientNetB7(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7')

        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(in_features=in_features, out_features=num_classes)
        # self.linear = nn.Linear(in_features = self.efficientnet._fc.out_features, out_features = num_classes)

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.efficientnet._fc.weight)
        stdv = 1 / math.sqrt(self.efficientnet._fc.in_features)
        self.efficientnet._fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.efficientnet(x)
        return x


from vit_pytorch import ViT


class Vit(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.v = ViT(
            image_size=256,
            # Image size. If you have rectangular images, make sure your image size is the maximum of the width and height
            patch_size=32,
            num_classes=num_classes,  # Number of classes to classify.
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.3,  # 0.1,
            emb_dropout=0.1
        )

    def forward(self, x):
        x = self.v(x)
        return x


# Pre trained ViT - B16 - weight freezed
# https://github.com/lukemelas/PyTorch-Pretrained-ViT#loading-pretrained-models
from pytorch_pretrained_vit import ViT


class ViTPretrainedFreeze(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = ViT('B_16_imagenet1k', pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        # initialize
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1 / math.sqrt(self.model.fc.in_features)
        self.model.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)


# VGG line has a problem with 13, 19 version
class Vgg11(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vgg11_bn(pretrained=True)

        # for param in self.model.parameters():
        #     param.requires_grad = False
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        # initialize
        torch.nn.init.xavier_uniform_(self.model.classifier[6].weight)
        stdv = 1 / math.sqrt(self.model.classifier[6].in_features)
        self.model.classifier[6].bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)


class Vgg11Freeze(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vgg11_bn(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        # initialize
        torch.nn.init.xavier_uniform_(self.model.classifier[6].weight)
        stdv = 1 / math.sqrt(self.model.classifier[6].in_features)
        self.model.classifier[6].bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)


class Vgg13(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vgg13_bn(pretrained=True)

        # for param in self.model.parameters():
        #     param.requires_grad = False
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        # initialize
        torch.nn.init.xavier_uniform_(self.model.classifier[6].weight)
        stdv = 1 / math.sqrt(self.model.classifier[6].in_features)
        self.model.classifier[6].bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)


class Vgg13Freeze(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vgg13_bn(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        # initialize
        torch.nn.init.xavier_uniform_(self.model.classifier[6].weight)
        stdv = 1 / math.sqrt(self.model.classifier[6].in_features)
        self.model.classifier[6].bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)


class Vgg16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vgg16_bn(pretrained=True)

        # for param in self.model.parameters():
        #     param.requires_grad = False
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        # initialize
        torch.nn.init.xavier_uniform_(self.model.classifier[6].weight)
        stdv = 1 / math.sqrt(self.model.classifier[6].in_features)
        self.model.classifier[6].bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)


class Vgg16Freeze(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vgg16_bn(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        # initialize
        torch.nn.init.xavier_uniform_(self.model.classifier[6].weight)
        stdv = 1 / math.sqrt(self.model.classifier[6].in_features)
        self.model.classifier[6].bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)


# ERROR
# TODO fix the model, input size should be 3x299x299
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#inception-v3
class Inception(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.inception_v3(pretrained=True)
        set_parameter_requires_grad(self.model, feature_extracting = False) # if True -> FREEZE # TODO 이 부분을 모듈화 해서 싹 다 추가시켜주자
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # Handle the auxilary net
        num_ftrs = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 299

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.model.AuxLogits.fc.weight)
        stdv = 1 / math.sqrt(self.model.AuxLogits.fc.in_features)
        self.model.AuxLogits.fc.bias.data.uniform_(-stdv, stdv)

        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1 / math.sqrt(self.model.fc.in_features)
        self.model.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)
        
class ResNet18Dropout(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # https://discuss.pytorch.org/t/inject-dropout-into-resnet-or-any-other-network/66322/3
        self.resnet18.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.25, training=m.training))        

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.resnet18.fc.weight)
        stdv = 1 / math.sqrt(self.resnet18.fc.in_features)
        self.resnet18.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.resnet18(x)
        return x

class ResNet18MSD(nn.Module):
    """_summary_
    Name: Multi-sample Dropout ResNet18-pretrained

    Args:
        nn (_type_): _description_
    
    self.dropout_p = 0.5 (default):  
        the percentile of each dropout layers
        
    self.dropout_n = 5 (default):  
        the number of dropout layers


    ref: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100961
    """
    def __init__(self, num_classes):
        super().__init__()
        self.dropout_p = 0.5
        self.dropout_n = 5
        self.resnet18 = models.resnet18(pretrained=True)
        self.dropouts = nn.ModuleList([
            nn.Dropout(self.dropout_p) for _ in range(self.dropout_n)
        ])
        in_features = self.resnet18.fc.out_features
        self.linear = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.linear.weight)
        stdv = 1 / math.sqrt(self.linear.in_features) 
        self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.resnet18(x)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.linear(dropout(x))
            else:
                h += self.linear(dropout(x))
        output = h / len(self.dropouts)
        return output


class ResNet18FreezeTop6(nn.Module):
    """
    For Overfitting
    ref. https://www.kaggle.com/sandhyakrishnan02/face-mask-detection-using-torch/notebook
    """
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes, bias=True)

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1 / math.sqrt(self.model.fc.in_features)
        self.model.fc.bias.data.uniform_(-stdv, stdv)

        # freeze top 6 layers
        for i, child in enumerate(self.model.children()):
            if i == 6 : break # break point
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet34FreezeTop6(nn.Module):
    """
    For Overfitting
    ref. https://www.kaggle.com/sandhyakrishnan02/face-mask-detection-using-torch/notebook
    """
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes, bias=True)

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1 / math.sqrt(self.model.fc.in_features)
        self.model.fc.bias.data.uniform_(-stdv, stdv)

        # freeze top 6 layers
        for i, child in enumerate(self.model.children()):
            if i == 6 : break # break point
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return x


class EfficientNetB0FreezeTop3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)

        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)
        # self.linear = nn.Linear(in_features = self.efficientnet._fc.out_features, out_features = num_classes)

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.efficientnet.classifier[1].weight)
        stdv = 1 / math.sqrt(self.efficientnet.classifier[1].in_features)
        self.efficientnet.classifier[1].bias.data.uniform_(-stdv, stdv)

        # Freeze Top 6 layers
        for i, child in enumerate(self.efficientnet.children()):
            if i == 6 : break # break point; Top 6 layers
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        x =  self.efficientnet(x)
        return x


class EfficientNetB4FreezeTop3(nn.Module):
    """_summary_
    size recommended to 380
    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = models.efficientnet_b4(pretrained=True)

        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)
        # self.linear = nn.Linear(in_features = self.efficientnet._fc.out_features, out_features = num_classes)

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.efficientnet.classifier[1].weight)
        stdv = 1 / math.sqrt(self.efficientnet.classifier[1].in_features)
        self.efficientnet.classifier[1].bias.data.uniform_(-stdv, stdv)

        # Freeze Top 6 layers
        for i, child in enumerate(self.efficientnet.features.children()):
            if i == 3 : break # break point; Top 6 layers
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        x =  self.efficientnet(x)
        return x

'''
class EfficientNetB3(nn.Module):
    """_summary_
    size recommended to 300
    """
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = models.efficientnet_b3(pretrained=True)

        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)
        # self.linear = nn.Linear(in_features = self.efficientnet._fc.out_features, out_features = num_classes)

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.efficientnet.classifier[1].weight)
        stdv = 1 / math.sqrt(self.efficientnet.classifier[1].in_features)
        self.efficientnet.classifier[1].bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x =  self.efficientnet(x)
        return x
'''

class EfficientNetB3MSD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dropout_p = 0.5
        self.dropout_n = 5
        self.efficientnet = models.efficientnet_b3(pretrained=True)

        self.dropouts = nn.ModuleList([
            nn.Dropout(self.dropout_p) for _ in range(self.dropout_n)
        ])
        
        in_features = self.efficientnet.classifier[1].out_features
        self.linear = nn.Linear(in_features=in_features, out_features=num_classes)

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.linear.weight)
        stdv = 1 / math.sqrt(self.linear.in_features)
        self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.efficientnet(x)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.linear(dropout(x))
            else:
                h += self.linear(dropout(x))
        output = h / len(self.dropouts)
        return output


class EfficientNetB7(nn.Module):
    """_summary_
    size recommended to 600
    """
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = models.efficientnet_b7(pretrained=True)

        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)
        # self.linear = nn.Linear(in_features = self.efficientnet._fc.out_features, out_features = num_classes)

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.efficientnet.classifier[1].weight)
        stdv = 1 / math.sqrt(self.efficientnet.classifier[1].in_features)
        self.efficientnet.classifier[1].bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x =  self.efficientnet(x)
        return x

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
