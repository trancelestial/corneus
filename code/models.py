from shutil import Error
from networks import SimpleConvNet, AlexLikeNet
from base.base_model import BaseModel
from torchvision.models import resnet18
import torch

class SimpleConv(BaseModel):
    def __init__(self, size_in, size_out, activation='relu', device='cuda', **kwargs):
        super().__init__(device=device)
        self.net = SimpleConvNet(size_in, size_out, activation, **kwargs)

class AlexNetLike(BaseModel):
    def __init__(self, size_in, size_out, device='cuda', **kwargs):
        super().__init__(device=device)
        self.net = AlexLikeNet(size_in, size_out, **kwargs)

class ResNet18(BaseModel):
    def __init__(self, size_in, size_out, pretrained=True, tune_only_last=False, device='cuda', **kwargs):
        super().__init__(device=device)
        if size_in != torch.Size((224, 224)):
            raise ValueError(f'Input size has to be 224x224')
        self.net = resnet18(pretrained=pretrained)

        if tune_only_last:
            for param in self.net.parameters():
                param.requires_grad = False

        self.net.fc = torch.nn.Linear(self.net.fc.in_features, size_out)

