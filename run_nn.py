import numpy as np
import pandas as pd
import datetime
import logging
import argparse
import json
import time
import sys

from utils import load_datasets, load_target
from utils.line_notification import send_message
from models.nn import MLPNet

from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')

s = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

now = datetime.datetime.now()
logging.basicConfig(
    filename='./logs/log_nn_{0:%Y%m%d%H%M%S}.log'.format(now), level=logging.DEBUG
)
logging.debug('./logs/log_nn_{0:%Y%m%d%H%M%S}.log'.format(now))

feats = config['features']
logging.debug(feats)

target_name = config['target_name']

X_train_all, X_test = load_datasets(feats)
y_train_all = load_target(target_name)
logging.debug(X_train_all.shape)

nn_params = config['nn_params']
logging.debug(nn_params)

batch_size = nn_params['batch_size']
num_epochs = nn_params['num_epochs']
learning_rate = nn_params['learning_rate']
device = nn_params['device']

net = MLPNet(X_train_all.shape[1]).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

metrics = []

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000)
oof = np.zeros(len(X_train_all))
predictions = np.zeros(len(X_test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_all, y_train_all)):
    print("===fold {}===".format(fold_ + 1))
    logging.debug("===fold {}===".format(fold_ + 1))
    x_train, x_val = X_train_all.iloc[trn_idx], X_train_all.iloc[val_idx]
    y_train, y_val = y_train_all.iloc[trn_idx], y_train_all.iloc[val_idx]

    batch_no = len(x_train) // batch_size

# NN ==========================================================================
    for epoch in range(num_epochs):

        x_train2, y_train2 = shuffle(x_train, y_train, random_state=epoch)

        net.train()
        for i in range(batch_no):
            start = i * batch_size
            end = start + batch_size
            x_var = Variable(torch.FloatTensor(x_train2.values[start:end]))
            t_var = Variable(torch.FloatTensor(y_train2.values[start:end])).view(-1, 1)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(x_var)
            loss = criterion(outputs, t_var)
            loss.backward()
            optimizer.step()

        net.eval()
        x_val_var = Variable(torch.FloatTensor(x_val.values), requires_grad=True)
        x_train_var = Variable(torch.FloatTensor(x_train.values), requires_grad=True)
        with torch.no_grad():
            x_train_outputs = net(x_train_var).numpy().reshape(x_train.shape[0])
            x_val_outputs = net(x_val_var).numpy().reshape(x_val.shape[0])
        train_auc = roc_auc_score(y_train, x_train_outputs)
        val_auc = roc_auc_score(y_val, x_val_outputs)

        if (epoch + 1) % 5 == 0:
            print("[%d]     training's auc: %.4f     valid_1's auc: %.4f" % (epoch + 1, train_auc, val_auc))
            logging.debug("[%d]     training's auc: %.4f     valid_1's auc: %.4f" % (epoch + 1, train_auc, val_auc))

    x_val_var = Variable(torch.FloatTensor(x_val.values), requires_grad=True)
    with torch.no_grad():
        outputs = net(x_val_var).numpy().reshape(x_val.shape[0])
    oof[val_idx] = outputs

    auc = roc_auc_score(y_val.values, outputs)
    metrics.append(round(auc, 5))
    print("valid's_loss: %.4f" % auc)
    logging.debug("valid's_loss: %.4f" % auc)

    x_test_var = Variable(torch.FloatTensor(X_test.values), requires_grad=True)
    with torch.no_grad():
        outputs = net(x_test_var)
    predictions += outputs.numpy().reshape(X_test.shape[0]) / folds.n_splits
# NN ==========================================================================

score = np.mean(metrics)

print("CV : {}".format(score))
logging.debug("CV : {}".format(score))

predictions = (predictions - min(predictions)) / (np.max(predictions) - np.min(predictions))

ID_name = config['ID_name']
sub = pd.DataFrame(pd.read_csv('./data/input/test.csv')[ID_name])

sub[target_name] = predictions

sub.to_csv(
    './data/output/sub_nn_{0:%Y%m%d%H%M%S}_{1}.csv'.format(now, score),
    index=False
)

e = time.time()

message = """{f}
cv: {cv:.4f}
scores: {s}
time: {t:.2f}[min]""".format(f=sys.argv[0], cv=score, s=metrics, t=(e - s) / 60)

send_message(message)
