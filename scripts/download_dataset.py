import os
from torchvision import datasets
from config import PROJECT_ROOT


data_dir = PROJECT_ROOT / "data"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

datasets.MNIST(
    root=str(data_dir),
    train=True,
    download=True,
)

datasets.MNIST(
    root=str(data_dir),
    train=False,
    download=True,
)