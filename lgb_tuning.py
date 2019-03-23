#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import argparse
import json

from utils import load_datasets, load_target

import optuna
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

now = datetime.datetime.now()
logging.basicConfig(
    filename='./logs/optuna_{0:%Y%m%d%H%M%S}.log'.format(now), level=logging.DEBUG
)
logging.debug('./logs/optuna_{0:%Y%m%d%H%M%S}.log'.format(now))

feats = config['features']
logging.debug(feats)

target_name = config['target_name']

X_train_all, X_test = load_datasets(feats)
y_train_all = load_target(target_name)
logging.debug(X_train_all.shape)


def objective(trial):
    train_x, test_x, train_y, test_y = train_test_split(X_train_all, y_train_all, test_size=0.20, random_state=4590)

    dtrain = lgb.Dataset(train_x, label=train_y)
    dval = lgb.Dataset(test_x, label=test_y)

    num_round = 10000
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'feature_fraction': trial.suggest_loguniform('feature_fraction', 0.8, 0.95),
        'bagging_fraction': trial.suggest_loguniform('bagging_fraction', 0.8, 0.95),
        'bagging_seed': 11,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': 0.09,
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 1000),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-2, 1.0)
    }

    gbm = lgb.train(param, dtrain, num_round, valid_sets=[dtrain, dval], verbose_eval=100, early_stopping_rounds=100)
    preds = gbm.predict(test_x)
    auc = roc_auc_score(test_y, preds)
    return 1 - auc


study = optuna.create_study()
study.optimize(objective, n_trials=20)

print('Number of finished trials: {}'.format(len(study.trials)))
logging.debug('Number of finished trials: {}'.format(len(study.trials)))

print('Best trial:')
trial = study.best_trial
logging.debug('Best trial:')
logging.debug(study.best_trial)

print('  Value: {}'.format(trial.value))
logging.debug('  Value: {}'.format(trial.value))

print('  Params: ')
logging.debug('  Params: ')
for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
        logging.debug('    {}: {}'.format(key, value))
