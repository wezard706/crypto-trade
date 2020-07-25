import os
import sys
import numpy as np
import talib as ta
from datetime import datetime, timedelta
import logging

from common.utils import calc_marubozu

root_module = os.path.basename(sys.argv[0])
if root_module == 'tr.py':
    logger = logging.getLogger('crypto')
elif root_module == 'sim.py':
    logger = logging.getLogger('simulate')
else:
    pass

class ModelBb:
    '''
    長期足: 長期のトレンドを見る -> エントリーの向きを決める
    中期足: 中期のトレンドとその始まり/中盤/終わりを判断する -> エントリーを決める
    短期足: 短期のトレンドを見る -> エントリーのタイミングを測る
    トレンド転換の見極め方法: ローソク足パターン or ローソク足パターン * 移動平均乖離率
    ローソク足パターン: 上昇 -> 上長ヒゲ陰線 or 大陰線、包み線、はらみ線、切り込み線、毛抜き底、前の高値と安値を切り下げる
    （参考: https://orange2.net/a_main/candle_bottom_patterns2/）
    '''
    def __init__(self, pair, candle_type_s, candle_type_m, candle_type_l, api):
        self.pair = pair
        self.candle_type_s = candle_type_s
        self.candle_type_m = candle_type_m
        self.candle_type_l = candle_type_l
        self.api = api

    def is_entry_long(self, candle):
        end_dt = datetime.fromtimestamp(candle[5] / 1000)
        start_dt = (end_dt - timedelta(1)).strftime('%Y-%m-%d %H:%M:%S')
        end_dt = end_dt.strftime('%Y-%m-%d %H:%M:%S')
        candles_s = self.api.get_candles(self.pair, candle_type='1min', start_dt=start_dt, end_dt=end_dt)
        candles_m = self.api.get_candles(self.pair, candle_type='5min', start_dt=start_dt, end_dt=end_dt)
        candles_l = self.api.get_candles(self.pair, candle_type='1hour', start_dt=start_dt, end_dt=end_dt)
        close_s = candles_s[:, 3]
        close_m = candles_m[:, 3]
        close_l = candles_l[:, 3]
        bb_s = ta.BBANDS(close_s, timeperiod=20, nbdevup=0.5, nbdevdn=0.5)
        bb_m = ta.BBANDS(close_m, timeperiod=20, nbdevup=0.5, nbdevdn=0.5)
        bb_l = ta.BBANDS(close_l, timeperiod=20, nbdevup=0.5, nbdevdn=0.5)
        # BBで売買タイミングを決定: https://f-pedia.jp/bollinger-band-1hour-5min-daytre/
        if self.check_trend_01(close_l, bb_l) == 'up': # 長期足でトレンド確認
            if self.check_trend_01(close_m, bb_m) == 'up': # 中期足でトレンド確認
                open_m = candles_m[:, 0]
                trend_change_01 = self.check_trend_change_01(open_m, close_m)
                trend_change_02 = self.check_trend_change_02(close_m, bb_m[1])
                if trend_change_01 == 'up' and trend_change_02 != 'up': # トレンド転換の有無を確認
                    # 短期足でタイミングを計る
                    # エントリーの優先順位
                    # 1. 下降トレンド -> エントリー
                    # 2. 上昇トレンド -> エントリー (下降トレンドを待つと機会を逃すため)
                    # 3. レンジ相場 -> 様子見
                    trend_s = self.check_trend_01(close_s, bb_s)
                    if (trend_s == 'up') or (trend_s == 'down'):
                        return True
        return False

    def is_entry_short(self, candle):
        end_dt = datetime.fromtimestamp(candle[5] / 1000)
        start_dt = (end_dt - timedelta(2)).strftime('%Y-%m-%d %H:%M:%S')
        end_dt = end_dt.strftime('%Y-%m-%d %H:%M:%S')
        candles_s = self.api.get_candles(self.pair, candle_type='1min', start_dt=start_dt, end_dt=end_dt)
        candles_m = self.api.get_candles(self.pair, candle_type='5min', start_dt=start_dt, end_dt=end_dt)
        candles_l = self.api.get_candles(self.pair, candle_type='1hour', start_dt=start_dt, end_dt=end_dt)
        close_s = np.array(candles_s).astype(np.float64)[:, 3]
        close_m = np.array(candles_m).astype(np.float64)[:, 3]
        close_l = np.array(candles_l).astype(np.float64)[:, 3]
        bb_s = ta.BBANDS(close_s, timeperiod=20, nbdevup=1, nbdevdn=1)
        bb_m = ta.BBANDS(close_m, timeperiod=20, nbdevup=1, nbdevdn=1)
        bb_l = ta.BBANDS(close_l, timeperiod=20, nbdevup=1, nbdevdn=1)

        # BBで売買タイミングを決定: https://f-pedia.jp/bollinger-band-1hour-5min-daytre/
        if self.check_trend_01(close_l, bb_l) == 'down':
            if self.check_trend_02(close_m, bb_m) == 'down':
                if self.check_trend_02(close_s, bb_s) == 'down':
                    return True
                elif self.check_trend_01(close_s, bb_s) == 'up':
                    return True
        return False

    def check_trend_01(self, close, bb):
        upper = bb[0]
        lower = bb[2]
        if close[-2] > upper[-2]:
            return 'up'
        elif close[-2] < lower[-2]:
            return 'down'
        else:
            return 'range'

    def check_trend_02(self, close, bb):
        middle = bb[1]
        if close[-2] > middle[-2]:
            return 'up'
        elif close[-2] < middle[-2]:
            return 'down'
        else:
            return 'range'

    def check_trend_change_01(self, open, close, duration=3):
        marubozu = calc_marubozu(open, close)
        target = marubozu[-duration:]
        if np.all(target == 0):
            return 'range'
        elif np.any(target == 100) and np.any(target == -100):
            return 'updown'
        elif np.any(target == 100):
            return 'up'
        elif np.any(target == -100):
            return 'down'

    def check_trend_change_02(self, close, ma):
        maer = ((close - ma) / ma) * 100  # moving average estrangement rate
        if maer[-2] > 2:
            return 'down'
        elif maer[-2] < 2:
            return 'up'
        else:
            return 'range'
