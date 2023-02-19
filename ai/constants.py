import os as _os
from pathlib import Path

PROJECT_FOLDER_PATH = Path(_os.path.abspath(__file__)).parent.parent.absolute()
MODEL_WEIGHTS_PATH = _os.path.join(PROJECT_FOLDER_PATH, "model_weights")
TRAINING_IMAGES_PATH = _os.path.join(PROJECT_FOLDER_PATH, "datasets/train-images-idx3-ubyte")
TRAINING_LABELS_PATH = _os.path.join(PROJECT_FOLDER_PATH, "datasets/train-labels-idx1-ubyte")
TEST_IMAGES_PATH = _os.path.join(PROJECT_FOLDER_PATH, "datasets/t10k-images-idx3-ubyte")
TEST_LABELS_PATH = _os.path.join(PROJECT_FOLDER_PATH, "datasets/t10k-labels-idx1-ubyte")