import math
import os
import sys
import numpy as np
import pandas as pd
import talib as ta
from datetime import datetime, timedelta
import logging
import pickle

from common.utils import calc_rocp_std, dict2str
from ml.train_regression import _extract_feature, get_params

root_module = os.path.basename(sys.argv[0])
if root_module == 'tr.py':
    logger = logging.getLogger('crypto')
elif root_module == 'sim.py':
    logger = logging.getLogger('simulate')
else:
    pass

class ModelLGB:
    def __init__(self, pair, api, loss_lower):
        self.pair = pair
        self.api = api
        self.loss_lower = loss_lower
        with open('ml/model/clf.pkl', mode='rb') as f:
            self.clf = pickle.load(f)
        with open('ml/model/clf_upper.pkl', mode='rb') as f:
            self.clf_upper = pickle.load(f)
        with open('ml/model/clf_lower.pkl', mode='rb') as f:
            self.clf_lower = pickle.load(f)

    def calc_entry_score(self, candle):
        end_dt = datetime.fromtimestamp(candle[5] / 1000)
        start_dt = (end_dt - timedelta(1)).strftime('%Y-%m-%d %H:%M:%S')
        end_dt = end_dt.strftime('%Y-%m-%d %H:%M:%S')

        features = None
        candle_types = ['1min', '5min', '15min', '30min']
        for candle_type in candle_types:
            candles = self.api.get_candles(self.pair, candle_type=candle_type, start_dt=start_dt, end_dt=end_dt)
            candles = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
            candles.index = candles.timestamp.map(lambda x: datetime.fromtimestamp(x / 1000))
            params = get_params('1min')
            feature = _extract_feature(candles, params)
            feature = feature.fillna(method='ffill')
            feature.columns = [c + '_' + candle_type for c in feature.columns]

            if features is None:
                features = feature
            else:
                features = pd.concat([features, feature], axis=1)
                features = features.fillna(method='ffill')

        # 予測値を低く見積もって（25%点）さらにROCPの標準偏差も考慮した値が０より大きければロング
        X = features[features.index == end_dt]
        y_pred = self.clf.predict(X)[0]
        y_pred_upper = self.clf_upper.predict(X)[0]
        y_pred_lower = self.clf_lower.predict(X)[0]
        rocp_std = calc_rocp_std(self.api, self.pair, start_dt, end_dt)
        score = y_pred_lower - rocp_std
        result = {'y_pred': y_pred, 'y_pred_upper': y_pred_upper, 'y_pred_lower': y_pred_lower, 'rocp_std': rocp_std, 'score': score}
        logger.debug(dict2str(result))
        if score > 0:
            return [True, result]
        return [False, result]

    def is_exit_long(self, price_now, price_bought, timestamp_now, timestamp_bought):
        reward = price_now - price_bought
        dt_now = datetime.fromtimestamp(timestamp_now / 1000)
        dt_bought = datetime.fromtimestamp(timestamp_bought / 1000)
        dt_delta = dt_now - dt_bought
        logger.debug('reward: {}, now: {}, bought: {}, delta: {}'.format(reward, dt_now, dt_bought, dt_delta))
        if (reward < self.loss_lower) or (dt_delta.total_seconds() >= 240):
            return True
        return False