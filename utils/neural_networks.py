from torch import nn
from config import LOSS_FN, OPTIMIZER_TYPE, LEARNING_RATE


def create_FC(architecture : list[int]) -> nn.Sequential:
    model = nn.Sequential()
    for i in range(len(architecture)-1):
        model.append(nn.Linear(in_features=architecture[i], out_features=architecture[i+1]))
        if i == len(architecture)-2:
            continue
        model.append(nn.ReLU())
    return model


def retrieve_models() -> list[dict]:
    models = list()
    architectures = [
        [784, 30, 15, 10],
        [784, 40, 20, 10],
        [784, 80, 40, 20, 10],
        [784, 100, 50, 25, 20, 10],
    ]

    for architecture in architectures:
        model :  nn.Sequential = create_FC(architecture)
        optimizer = OPTIMIZER_TYPE(model.parameters(), lr=LEARNING_RATE)
        models.append(
            {
                'model' : model,
                'loss_fn' : LOSS_FN,
                'optimizer' : optimizer
            }
        )
    return models

