import os
import sys
import numpy as np
import math
import talib as ta
from datetime import datetime, timedelta
import logging

root_module = os.path.basename(sys.argv[0])
if root_module == 'tr.py':
    logger = logging.getLogger('crypto')
elif root_module == 'sim.py':
    logger = logging.getLogger('simulate')
else:
    pass

class ModelDow:
    def __init__(self, pair, candle_type, timeperiod_s, timeperiod_m, timeperiod_l, timeperiod_xl, api):
        self.pair = pair
        self.candle_type = candle_type
        self.timeperiod_s = timeperiod_s
        self.timeperiod_m = timeperiod_m
        self.timeperiod_l = timeperiod_l
        self.timeperiod_xl = timeperiod_xl
        self.api = api

    def is_entry_long(self, candle):
        end_dt = datetime.fromtimestamp(candle[5] / 1000)
        start_dt = end_dt - timedelta(2)
        candles = self.api.get_candles(self.pair, self.candle_type, start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S'))
        close = np.array(candles).astype(np.float64)[:, 3]
        ma_s = ta.SMA(close, self.timeperiod_s)
        ma_m = ta.SMA(close, self.timeperiod_m)
        ma_l = ta.SMA(close, self.timeperiod_l)
        ma_xl = ta.SMA(close, self.timeperiod_xl)
        if (self.check_trend_03(ma_xl, step=3, threshold=0.15) == 'up'):
            if (self.check_trend_03(ma_s) == 'up') and (self.check_trend_03(ma_m) == 'up') and (self.check_trend_03(ma_l) == 'up'):
                return True
        return False

    def is_entry_short(self, candle):
        end_dt = datetime.fromtimestamp(candle[5] / 1000)
        start_dt = end_dt - timedelta(2)
        candles = self.api.get_candles(self.pair, self.candle_type, start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S'))
        close = np.array(candles).astype(np.float64)[:, 3]
        ma_s = ta.SMA(close, self.timeperiod_s)
        ma_m = ta.SMA(close, self.timeperiod_m)
        ma_l = ta.SMA(close, self.timeperiod_l)
        ma_xl = ta.SMA(close, self.timeperiod_xl)
        if (self.check_trend_03(ma_xl, step=3, threshold=0.15) == 'down'):
            if (self.check_trend_03(ma_s) == 'down') and (self.check_trend_03(ma_m) == 'down') and (self.check_trend_03(ma_l) == 'down'):
                return True
        return False

    def check_trend_01(self, ma_s, ma_m):
        '''
        短期移動平均 > 中期移動平均
        '''
        if ma_s[-1] > ma_m[-1]:
            return True
        return False

    def check_trend_02(self, candle, ma):
        low = float(candle[2])
        high = float(candle[1])
        if low > ma[-1]:  # 低値 > 中期移動平均
            return 'up'
        elif high < ma[-1]:  # 高値 < 中期移動平均
            return 'down'
        else:
            return 'range'

    def check_trend_03(self, ma, step=1, threshold=0.0):
        '''
        移動平均の角度
        '''
        degree = math.degrees(math.atan((ma[-1] - ma[-(step + 1)])))
        if degree > threshold:
            return 'up'
        elif degree < -threshold:
            return 'down'
        else:
            return 'range'

    def check_trend_04(self, ma_s, ma_m, ma_l):
        '''
        長期 < 中期 < 短期移動平均
        '''
        if (ma_l[-1] < ma_m[-1]) and (ma_m[-1] < ma_s[-1]):
            return 'up'
        elif (ma_l[-1] > ma_m[-1]) and (ma_m[-1] > ma_s[-1]):
            return 'down'
        else:
            return 'range'
