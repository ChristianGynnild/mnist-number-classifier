import torch
from torch import nn

class LinearNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 718),
            nn.ReLU(),
            nn.Linear(718, 10),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Flatten(),
        )
        self.linear_relu_model = nn.Sequential(
            nn.Linear(576,256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 84, bias=False),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(84, 10),
            nn.Softmax()            
        )
    

    def forward(self, x):
        x = self.cnn_model(x)
        #x = x.view(-1, 16*5*5)
        x = self.linear_relu_model(x)
        return x
