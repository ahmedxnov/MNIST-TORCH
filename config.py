from pathlib import Path
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent

# Loss function (stateless)
LOSS_FN = nn.CrossEntropyLoss()

# Optimizer settings
OPTIMIZER_TYPE = optim.Adam
LEARNING_RATE = 0.01