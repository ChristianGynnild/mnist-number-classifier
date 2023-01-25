import torch
from torch import nn
import dataset
import numpy as np

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



def train(images, labels, model, loss_fn, optimizer):
    amount_batches = images.shape[0]
    batch_size = images.shape[1]
    size = amount_batches*batch_size

    model.train()
    for batch, (x, y) in enumerate(zip(images, labels)):
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


def test(images, labels, model, loss_fn):
    amount_batches = images.shape[0]
    batch_size = images.shape[1]
    size = amount_batches*batch_size

    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in zip(images, labels):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= amount_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



def train_and_test():
    load_images = lambda filepath: torch.from_numpy(dataset.to_batches(dataset.load_images(filepath)))        
    load_labels = lambda filepath: torch.from_numpy(dataset.to_batches(dataset.load_labels(filepath)))        

    images_training = load_images("datasets/train-images-idx3-ubyte")
    images_test =     load_images("datasets/t10k-images-idx3-ubyte")
    labels_training = load_labels("datasets/train-labels-idx1-ubyte")
    labels_test =     load_labels("datasets/t10k-labels-idx1-ubyte")

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(images_training, labels_training, model, loss_fn, optimizer)
        test(images_test, labels_test, model, loss_fn)



