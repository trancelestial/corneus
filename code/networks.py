import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_net import BaseNet

class SimpleConv(BaseNet):
    def __init__(self, size_in, size_out, activation='relu'):
        super().__init__(size_in, size_out)
        valid_activations = {'relu': F.relu, 'leaky': F.leaky_relu, 'tanh': F.tanh,
                             'sigmoid': F.sigmoid}
        assert activation in valid_activations, f'Invalid activation function: {valid_activations}!'
        self.activation = valid_activations[activation]
        self.size_in = size_in

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.flat_feat_no = size_in[0]//(2**5) * size_in[1]//(2**5) * 32
        self.fc1 = nn.Linear(self.flat_feat_no, 256)
        self.fc2 = nn.Linear(256, size_out)

    def forward(self, x):
        x = F.max_pool2d(self.activation(self.conv1(x)), (2,2))
        x = F.max_pool2d(self.activation(self.conv2(x)), (2,2))
        x = F.max_pool2d(self.activation(self.conv3(x)), (2,2))
        x = F.max_pool2d(self.activation(self.conv4(x)), (2,2))
        x = F.max_pool2d(self.activation(self.conv5(x)), (2,2))
        x = x.view(-1, self.flat_feat_no)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x

class AlexLike(BaseNet):
    def __init__(self, size_in, size_out):
        super().__init__(size_in, size_out)
        self.size_in = size_in

        self.feat = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 384, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.cls = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, size_out),
        )

    def forward(self, x):
        x = self.feat(x)
        x = self.avgpool(x)
        x = x.view(-1, 256*6*6)
        x = self.cls(x)

        return x

