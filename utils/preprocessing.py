import numpy as np
from numpy.typing import NDArray

def load_mnist_images(path : str) -> NDArray[np.uint8]:
    return np.fromfile(path, dtype=np.uint8, offset=16)

def load_mnist_labels(path: str) -> NDArray[np.uint8]:
    return np.fromfile(path, dtype=np.uint8, offset=8)

def preprocess(images : NDArray[np.uint8]) -> None:
    images.shape = (-1, 784)
    images_float = images.astype(np.float32)
    images[:] = images_float / 255.0