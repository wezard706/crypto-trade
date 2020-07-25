import math
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import pickle

from common.utils import dict2str
from ml.train_binary import extract_feature

root_module = os.path.basename(sys.argv[0])
if root_module == 'tr.py':
    logger = logging.getLogger('crypto')
elif root_module == 'sim.py':
    logger = logging.getLogger('simulate')
else:
    pass

class ModelLGBBinary:
    def __init__(self, pair, api, norm_mean, norm_std, lower, upper, loss_lower):
        self.pair = pair
        self.api = api
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.lower = lower
        self.upper = upper
        self.loss_lower = loss_lower
        with open('ml/model/201901010000_201902021759_clf_binary.pkl', mode='rb') as f:
        #with open('ml/model/201901250000_201902021745_clf_binary.pkl', mode='rb') as f:
            self.clf = pickle.load(f)

    def calc_score(self, candle):
        start_dt = datetime.fromtimestamp(candle[5] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        end_dt = start_dt

        logger.debug('extract feature from {} to {}'.format(start_dt, end_dt))
        X = extract_feature(start_dt, end_dt, self.pair, candle_types=['1min', '5min'])
        y_pred = self.clf.predict_proba(X)[0, 1]
        score = (y_pred - self.norm_mean) / self.norm_std

        X.to_csv('X_{}.csv'.format(end_dt))
        pd.Series(y_pred).to_csv('y_pred_{}.csv'.format(end_dt))

        if score > self.upper:
            return ['buy', score]
        elif score < self.lower:
            return ['sell', score]
        else:
            return ['wait', score]

    def is_cut_loss(self, price_now, price_bought):        
        if (float(price_now) - float(price_bought)) < self.loss_lower:
            return True
        else:
            return False