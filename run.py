#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import datetime
import logging
from sklearn.model_selection import StratifiedKFold
import argparse
import json
import time
import sys

from utils import load_datasets, load_target, send_line_notification
from logs.logger import log_best
from models.lgbm import train_and_predict
from scripts.save_importances import save
from scripts.sampling import downsampling

import warnings
warnings.filterwarnings('ignore')

s = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

now = datetime.datetime.now()
filename = 'log_lgbm_{0:%Y%m%d%H%M%S}.log'.format(now)
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

lgbm_params = config['lgbm_params']

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_all, y_train_all.values)):
    print("===fold {}===".format(fold_ + 1))
    logging.debug("===fold {}===".format(fold_ + 1))
    X_train, X_valid = (
        X_train_all.iloc[trn_idx, :], X_train_all.iloc[val_idx, :]
    )
    y_train, y_valid = y_train_all[trn_idx], y_train_all[val_idx]

    X_train, y_train = downsampling(X_train, y_train, fold_)

    y_pred, model = train_and_predict(
        X_train, X_valid, y_train, y_valid, X_test, lgbm_params
    )

    y_preds.append(y_pred)
    models.append(model)

    log_best(model, config['loss'])

try:
    save(models, X_train.columns, now)
    print("===Saved importances===")
    logging.debug('===Saved importances===')
except:
    pass

scores = [
    m.best_score['valid_1'][config['loss']] for m in models
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
    './data/output/sub_lgbm_{0:%Y%m%d%H%M%S}_{1}.csv'.format(now, score),
    index=False
)

e = time.time()

message = """
{f}
cv: {cv}
scores: {s}
time: {t} s
""".format(f=sys.argv[0], cv=score, s=scores, t=e-s)

send_line_notification(message)
