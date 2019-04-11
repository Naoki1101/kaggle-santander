#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime
import logging
from sklearn.model_selection import StratifiedKFold
import argparse
import json
import time
import sys

from utils import load_datasets, load_target
from utils.line_notification import send_message
from models.lr import lr_train_and_predict, save_coef, save_features
from scripts.sampling import downsampling

import warnings
warnings.filterwarnings('ignore')

s = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

now = datetime.datetime.now()
filename = 'log_lr_{0:%Y%m%d%H%M%S}.log'.format(now)
logging.basicConfig(
    filename='./logs/' + filename, level=logging.DEBUG
)
logging.debug('./logs/' + filename)

feats = config['features']
logging.debug(feats)

target_name = config['target_name']

X_train_all, X_test = load_datasets(feats)
y_train_all = load_target(target_name)
logging.debug(X_train_all.shape)

y_preds = []
models = []
scores = []
oof = np.zeros(len(X_train_all))

lr_params = config['lr_params']

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_all, y_train_all.values)):
    print("===fold {}===".format(fold_ + 1))
    logging.debug("===fold {}===".format(fold_ + 1))
    X_train, X_valid = (
        X_train_all.iloc[trn_idx, :], X_train_all.iloc[val_idx, :]
    )
    y_train, y_valid = y_train_all[trn_idx], y_train_all[val_idx]

    # print("*DOWN SAMPLING*")
    # logging.debug("*DOWN SAMPLING*")
    # X_train, y_train = downsampling(X_train, y_train, fold_)

    y_pred, model, val_auc, oof = lr_train_and_predict(
        X_train, X_valid, y_train, y_valid, X_test, lr_params, oof
    )

    print("valid_1's auc: %.4f" % val_auc)
    logging.debug("valid_1's auc: %.4f" % val_auc)

    y_preds.append(y_pred)
    models.append(model)
    scores.append(round(val_auc, 5))

print("===Save coef===")
logging.debug('===Save coef===')
save_coef(models, X_train.columns, now)

score = sum(scores) / len(scores)
print('===CV scores===')
print(scores)
print(score)
logging.debug('===CV scores===')
logging.debug(scores)
logging.debug(score)

ID_name = config['ID_name']
sub = pd.DataFrame(pd.read_csv('./data/input/test.csv')[ID_name])

y_sub = sum(y_preds) / len(y_preds)

print("=== Save features ===")
logging.debug("=== Save features ===")
save_features(oof, y_sub)

sub[target_name] = y_sub

sub.to_csv(
    './data/output/sub_lr_{0:%Y%m%d%H%M%S}_{1}.csv'.format(now, score),
    index=False
)

e = time.time()

message = """{f}
cv: {cv:.4f}
scores: {s}
time: {t:.2f}[min]""".format(f=sys.argv[0], cv=score, s=scores, t=(e - s) / 60)

send_message(message)
