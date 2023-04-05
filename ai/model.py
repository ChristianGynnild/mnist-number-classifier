import torch
from torch import nn
import numpy as np
from PIL import Image

from . import dataset
from .constants import MODEL_WEIGHTS_PATH, TRAINING_IMAGES_PATH, TRAINING_LABELS_PATH, TEST_IMAGES_PATH, TEST_LABELS_PATH
from .model_architectures import ConvolutionalNetwork as NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

def training_step(model, features, labels, loss_function, optimizer):
    batches_amount = features.shape[0]
    batch_size = features.shape[1]
    dataset_size = batches_amount*batch_size

    model.train()
    for batch, (x, y) in enumerate(zip(features, labels)):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x, y = x.to(device), y.to(device)

        # Compute prediction error        
        prediction = model(x)
        loss = loss_function(prediction, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{dataset_size:>5d}]")


def test(model, features, labels, loss_function):
    batches_amount = features.shape[0]
    batch_size = features.shape[1]
    dataset_size = batches_amount*batch_size

    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in zip(features, labels):
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            test_loss += loss_function(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= batches_amount
    correct /= dataset_size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def load_model():
    model = NeuralNetwork().to(device)
    
    try:model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device(device)))
    except Exception as e:print(f"Failed to load model weights. Exception:{e}")

    return model

def train(epochs=5):
    load_images = lambda filepath: torch.from_numpy(dataset.to_batches(np.array(list(map(dataset.image_preprocessing, dataset.load_images(filepath))))))     
    load_labels = lambda filepath: torch.from_numpy(dataset.to_batches(dataset.load_labels(filepath)))

    training_images = load_images(TRAINING_IMAGES_PATH)
    training_labels = load_labels(TRAINING_LABELS_PATH)
    test_images =     load_images(TEST_IMAGES_PATH)
    test_labels =     load_labels(TEST_LABELS_PATH)

    model = load_model()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        training_step(model, training_images, training_labels, loss_function, optimizer)
        test(model, test_images, test_labels, loss_function)
        if (t+1)%5 == 0:
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
            print(f"Saved PyTorch Model State to {MODEL_WEIGHTS_PATH}")

    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    print(f"Saved PyTorch Model State to {MODEL_WEIGHTS_PATH}")

def predict(model, image):
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

    array = dataset.image_preprocessing(np.array(image, dtype=np.float32).reshape((1, 1, 28,28)))

    model.eval()
    prediction = model(torch.from_numpy(array))
    prediction = int(prediction[0].argmax(0))
    print(prediction)
    return prediction