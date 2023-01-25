import torch
from torch import nn
import dataset
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

MODEL_WEIGHTS_PATH = "model_weights"


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



def train_and_test(epochs=5):
    load_images = lambda filepath: torch.from_numpy(dataset.to_batches(dataset.load_images(filepath)))        
    load_labels = lambda filepath: torch.from_numpy(dataset.to_batches(dataset.load_labels(filepath)))        

    images_training = load_images("datasets/train-images-idx3-ubyte")
    images_test =     load_images("datasets/t10k-images-idx3-ubyte")
    labels_training = load_labels("datasets/train-labels-idx1-ubyte")
    labels_test =     load_labels("datasets/t10k-labels-idx1-ubyte")

    model = NeuralNetwork().to(device)
    print(model)
    try:model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    except Exception:pass
    

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(images_training, labels_training, model, loss_fn, optimizer)
        test(images_test, labels_test, model, loss_fn)
        if (t+1)%5 == 0:
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
            print(f"Saved PyTorch Model State to {MODEL_WEIGHTS_PATH}")

    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    print(f"Saved PyTorch Model State to {MODEL_WEIGHTS_PATH}")

def predict(image_path):

    image = Image.open(image_path)
    print("Opened:", image_path)
    print("shape:",image.size)
    width_height_difference = image.width-image.height

    if width_height_difference<0:
        y_shift=int((image.height-image.width)/2)
        box = (0, y_shift, image.width, y_shift+image.width)
        print("Image is not rectangular, cropping the height")
        image = image.crop(box)
        print("Cropped size:", image.size)
    if width_height_difference>0:
        x_shift=int((image.width-image.height)/2)
        box = (x_shift, 0, x_shift+image.height, image.height)
        print("Image is not rectangular, cropping the width")
        image = image.crop(box)
        print("Cropped size:", image.size)

    size = 28, 28
    if image.size != size:
        image.thumbnail(size, Image.Resampling.LANCZOS)
        print("Downsampled image to:",image.size)

    image = image.convert('L') # Turn the picture grayscale

    array = np.array(image, dtype=np.float32).reshape((1,28,28))


    model = NeuralNetwork().to(device)
    try:model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    except Exception:pass
    model.eval()
    prediction = model(torch.from_numpy(array))
    print(int(prediction[0].argmax(0)))
    image.save("twst.jpeg", "png")