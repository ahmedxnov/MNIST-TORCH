from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch import cuda

PROJECT_ROOT = Path(__file__).resolve().parent

# Loss function (stateless)
LOSS_FN = nn.CrossEntropyLoss()

# Optimizer settings
OPTIMIZER_TYPE = optim.Adam
LEARNING_RATE = 0.01

#Training Settings
BATCH_SIZE = 64
EPOCHS = 10
DEVICE = "cuda" if cuda.is_available() else "cpu"