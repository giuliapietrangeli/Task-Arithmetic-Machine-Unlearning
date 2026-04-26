import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

# ResNet
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

from torchvision.models import vgg11, VGG11_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import vgg11_bn 

# VGG con bathc norm. 
class VGG11_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11_CIFAR10, self).__init__()
        self.model = vgg11_bn(weights=None) 
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# MobileNet
class MobileNet_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNet_CIFAR10, self).__init__()
        self.model = mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)