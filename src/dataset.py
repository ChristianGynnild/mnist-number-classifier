from binascii import Error
from PIL import Image
import numpy as np
import pickle
import os
import hashlib
import inspect
from filecache import filecache
from cli import cli

@filecache
def load_images(file):
  with open(file, "rb") as f:
    read_bytes = lambda byte_size:int.from_bytes(f.read(byte_size), byteorder="big")
    f.read(3) #Jump over first 3 bytes
    dimensions = read_bytes(1)
    images_amount = read_bytes(4)
    rows_amount = read_bytes(4)
    colums_amount = read_bytes(4)

    print("dimensions:", dimensions)
    print("shape:", (images_amount, rows_amount, colums_amount))

    images = []

    for i in range(images_amount):
      image = []
      for x in range(rows_amount):
        for y in range(colums_amount):
          image.append(read_bytes(1))
      images.append(image)

    images = np.array(images, dtype=np.uint8).reshape((images_amount, 28,28))
  return images


def load_labels(file):
  with open(file, "rb") as f:
    read_bytes = lambda byte_size:int.from_bytes(f.read(byte_size), byteorder="big")


    


def image_to_file(image, filename):
  image = Image.fromarray(image, 'L')
  image.save(filename)
  image.show()







