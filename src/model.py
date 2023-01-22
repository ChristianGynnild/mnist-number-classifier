import torch
from torch import nn
import dataset
import numpy as np

images_training = dataset.load_images("datasets/train-images-idx3-ubyte")
labels_training = dataset.load_labels("datasets/train-labels-idx1-ubyte")

images_test = dataset.load_images("datasets/t10k-images-idx3-ubyte")
labels_test = dataset.load_labels("datasets/t10k-labels-idx1-ubyte")


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(images, labels, model, loss_fn, optimizer):
    size = len(images)
    model.train()
    for batch, (x, y) in zip(images, labels):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


print(labels_test.shape)
print(images_test.shape)
print(images_training.shape)
print(labels_training.shape)

