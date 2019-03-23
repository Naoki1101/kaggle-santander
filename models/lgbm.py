import lightgbm as lgb
import logging

from logs.logger import log_evaluation


def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, lgbm_params):


    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    logging.debug(lgbm_params)


    logger = logging.getLogger('main')
    callbacks = [log_evaluation(logger, period=100)]


    num_round = 20000
    model = lgb.train(
        lgbm_params,
        lgb_train,
        num_boost_round=num_round,
        valid_sets=[lgb_train, lgb_eval],
        verbose_eval=200,
        early_stopping_rounds=200,
        callbacks=callbacks
    )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    return y_pred, model
