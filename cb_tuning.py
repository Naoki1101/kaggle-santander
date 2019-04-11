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
from catboost import Pool, CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

now = datetime.datetime.now()
logging.basicConfig(
    filename='./logs/optuna_cb_{0:%Y%m%d%H%M%S}.log'.format(now), level=logging.DEBUG
)
logging.debug('./logs/optuna_cb_{0:%Y%m%d%H%M%S}.log'.format(now))

feats = config['features'] + config['stacking_feats']
logging.debug(feats)

target_name = config['target_name']

X_train_all, X_test = load_datasets(feats)
y_train_all = load_target(target_name)
logging.debug(X_train_all.shape)


def objective(trial):
    train_x, valid_x, train_y, valid_y = train_test_split(X_train_all, y_train_all, test_size=0.20, random_state=4590)

    cb_train = Pool(train_x, label=train_y)

    num_round = 10000
    params = {
        'learning_rate': 0.05,
        'iterations': 5000,
        'depth': trial.suggest_int('depth', 4, 10),
        'random_strength': trial.suggest_int('random_strength', 0, 100),
        'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        'od_wait': trial.suggest_int('od_wait', 10, 50)
    }

    model = CatBoostClassifier(**params)
    model.fit(cb_train, plot=False, verbose_eval=200)

    pred = model.predict_proba(valid_x)[:, 1]
    auc = roc_auc_score(valid_y, pred)

    return 1 - auc


study = optuna.create_study()
study.optimize(objective, n_trials=20)

print('Number of finished trials: {}'.format(len(study.trials)))
logging.debug('Number of finished trials: {}'.format(len(study.trials)))

print('Best trial:')
trial = study.best_trial
logging.debug('Best trial:')
logging.debug(study.best_trial)

print('  Value: {}'.format(1 - trial.value))
logging.debug('  Value: {}'.format(1 - trial.value))

print('  Params: ')
logging.debug('  Params: ')
for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
        logging.debug('    {}: {}'.format(key, value))
