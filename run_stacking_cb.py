#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime
import logging
from sklearn.model_selection import StratifiedKFold, KFold
import argparse
import json
import time
import sys

from utils import load_datasets, load_target
from utils.line_notification import send_message
from logs.logger import cb_log_best
from models.cb import train_and_predict, save_importances, save_features
from scripts.sampling import downsampling
from features.base import get_arguments, generate_features

import warnings
warnings.filterwarnings('ignore')

s = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

now = datetime.datetime.now()
filename = 'log_stacking_cb_{0:%Y%m%d%H%M%S}.log'.format(now)
logging.basicConfig(
    filename='./logs/' + filename, level=logging.DEBUG
)
logging.debug('./logs/' + filename)

feats = config['features'] + config['stacking_feats']
# feats = config['stacking_feats']
logging.debug(feats)

target_name = config['target_name']

X_train_all, X_test = load_datasets(feats)
y_train_all = load_target(target_name)
logging.debug(X_train_all.shape)

y_preds = []
models = []
oof = np.zeros(len(X_train_all))

st_params = config['st_cb_params']

# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
folds = KFold(n_splits=5, random_state=4590)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_all, y_train_all.values)):
# for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_all)):
    print("===fold {}===".format(fold_ + 1))
    logging.debug("===fold {}===".format(fold_ + 1))
    X_train, X_valid = (
        X_train_all.iloc[trn_idx, :], X_train_all.iloc[val_idx, :]
    )
    y_train, y_valid = y_train_all[trn_idx], y_train_all[val_idx]

    print("*DOWN SAMPLING*")
    logging.debug("*DOWN SAMPLING*")
    X_train, y_train = downsampling(X_train, y_train, fold_)

    y_pred, model, oof = train_and_predict(
        X_train, X_valid, y_train, y_valid, X_test, st_params, oof
    )

    y_preds.append(y_pred)
    models.append(model)

    cb_log_best(model, config['loss'].upper())

# print("===Save importances===")
# logging.debug('===Save importances===')
# save_importances(models, feats, now)

scores = [
    round(m.best_score_['validation_0'][config['loss'].upper()], 5) for m in models
]
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

sub[target_name] = y_sub

sub.to_csv(
    './data/output/sub_stacking_cb_{0:%Y%m%d%H%M%S}_{1}.csv'.format(now, score),
    index=False
)

e = time.time()

message = """{f}
cv: {cv:.4f}
scores: {s}
time: {t:.2f}[min]""".format(f=sys.argv[0], cv=score, s=scores, t=(e - s) / 60)

send_message(message)
