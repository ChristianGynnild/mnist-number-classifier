from binascii import Error
from PIL import Image
import numpy as np
import pickle
import os
import hashlib
import inspect
from filecache import filecache
from cli import cli
from tqdm import tqdm

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

    images = np.array(images, dtype=np.uint8).reshape((images_amount, 28,28))
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


    


def image_to_file(image, filename):
  image = Image.fromarray(image, 'L')
  image.save(filename)
  image.show()






