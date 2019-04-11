import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from pathlib import Path
import random

random.seed(1000)

class MLPNet(nn.Module):
    def __init__(self, input_size, dropout=0.5):
        super(MLPNet, self).__init__()
        # self.fc1 = nn.Linear(input_size, 500)
        # self.fc2 = nn.Linear(500, 200)
        # self.fc3 = nn.Linear(200, 200)
        # self.fc4 = nn.Linear(200, 1)
        # self.dropout1 = nn.Dropout2d(dropout)
        # self.dropout2 = nn.Dropout2d(dropout)
        # self.dropout3 = nn.Dropout2d(dropout)

        self.fc1 = nn.Linear(input_size, 2000)
        self.fc2 = nn.Linear(2000, 1)
        self.dropout1 = nn.Dropout2d(dropout)

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.dropout1(x)
        # x = self.fc2(x)
        # x = self.dropout2(x)
        # x = self.fc3(x)
        # x = self.dropout3(x)
        # return self.fc4(x))

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        return self.fc2(x)



def save_features(oof, pred, overwrite=False):
    root = Path('./features/')
    train_path = root.joinpath('nn_train.feather')
    test_path = root.joinpath('nn_test.feather')
    if train_path.exists() and test_path.exists() and not overwrite:
        print('NN was skipped')
    else:
        pd.DataFrame(oof, columns=['nn_pred']).to_feather(str(train_path))
        pd.DataFrame(pred, columns=['nn_pred']).to_feather(str(test_path))


def scaling(data1, data2):
    if np.min(data1) <= np.min(data2):
        min_ = np.min(data1)
    else:
        min_ = np.min(data2)

    if np.max(data1) >= np.max(data2):
        max_ = np.max(data1)
    else:
        max_ = np.max(data2)

    return (data1 - min_) / (max_ - min_), (data2 - min_) / (max_ - min_)
