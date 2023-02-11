from PIL import Image
import numpy as np
from filecache import filecache
from tqdm import tqdm
from torchvision.transforms import ToTensor
import torch

@filecache
def load_images(file):
    with open(file, "rb") as f:
        read_bytes = lambda byte_size:int.from_bytes(f.read(byte_size), byteorder="big")
        f.read(3) #Jump over first 3 bytes
        dimensions = read_bytes(1)
        images_amount = read_bytes(4)
        rows_amount = read_bytes(4)
        colums_amount = read_bytes(4)

        print(f"Loading images from:{file}")
        print("dimensions:", dimensions)
        print("shape:", (images_amount, rows_amount, colums_amount))

        images = []

        for i in tqdm(range(images_amount)):
            image = []
            for x in range(rows_amount):
                for y in range(colums_amount):
                    image.append(read_bytes(1))
            images.append(image)

        images = np.array(images, dtype=np.float32).reshape((images_amount, 28,28))
    
    #Useing standard deviation to make images sharper
    images = images/255
    mean_px = images.mean().astype(np.float32)
    std_px = images.std().astype(np.float32)
    images = (images - mean_px)/(std_px)
    images = images*255

    return images

@filecache
def load_labels(file):
    with open(file, "rb") as f:
        read_bytes = lambda byte_size:int.from_bytes(f.read(byte_size), byteorder="big")
        f.read(3) #Jump over first 3 bytes
        dimensions = read_bytes(1)
        labels_amount = read_bytes(4)

        print(f"Loading labels from:{file}")
        print("dimensions:", dimensions)
        print("labels_amount:", (labels_amount))

        labels = []

        for i in tqdm(range(labels_amount)):
            label = read_bytes(1)
            labels.append(label)

        labels = np.array(labels, dtype=np.uint8)

    
    return labels

def to_batches(array, batch_size=64):
    for i in range(len(array)%batch_size):
        array = np.delete(array, -1,  axis=0)
    array = np.array(np.split(array, len(array)/batch_size))

    return array



def image_to_file(image, filename):
    image = Image.fromarray(image)
    image = image.convert("RGB")
    image.save(filename)
    image.show()