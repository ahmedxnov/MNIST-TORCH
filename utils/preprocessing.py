import numpy as np
from numpy.typing import NDArray

def load_mnist_images(path : str) -> NDArray[np.uint8]:
    return np.fromfile(path, dtype=np.uint8, offset=16)

def load_mnist_labels(path: str) -> NDArray[np.uint8]:
    return np.fromfile(path, dtype=np.uint8, offset=8)

def preprocess(images : NDArray[np.uint8]) -> NDArray[np.float32]:
    images = images.reshape(-1, 784)
    images = images.astype(np.float32)
    images = images / 255.0
    return images