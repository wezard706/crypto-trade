import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import calendar

from trade.api import API
from common.utils import str2timestamp, round_time

class ResourceManager:
    def __init__(self, messages, tickers, assets, tm):
        self.messages = messages
        start_dt = datetime.fromtimestamp(messages['1min'].iloc[0, 5] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        end_dt = datetime.fromtimestamp(messages['1min'].iloc[-1, 5] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        self.messages_summary = self.summary_messages(start_dt, end_dt)
        self.tickers = tickers
        self.assets = assets
        self.tm = tm

    def get_ticker(self):
        now = self.tm.get_timestamp()
        ticker = self.tickers[self.tickers.index == now]
        if ticker is None:
            print('There is no ticker which match to given timestamp')
            return -1
        return {'buy': ticker['buy'].values, 'sell': ticker['sell'].values}

    def summary_messages(self, start_dt, end_dt):
        messages_summary = {}

        pair = 'xrp_jpy'
        candle_types = ['1min', '5min', '15min', '30min', '1hour']
        for candle_type in candle_types:
            candles = API().get_candles(pair, candle_type, start_dt, end_dt)
            candles = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
            candles.index = candles['timestamp']
            messages_summary[candle_type] = candles
        return messages_summary

    def get_candle_status(self, target_dt, current_dt, candle_type):
        if candle_type == '1min':
            delta = 60
        elif candle_type == '5min':
            delta = 60 * 5
        elif candle_type == '15min':
            delta = 60 * 15
        elif candle_type == '30min':
            delta = 60 * 30
        elif candle_type == '1hour':
            delta = 60 * 60
        elif candle_type == '4hour':
            delta = 60 * 60 * 4
        elif candle_type == '8hour':
            delta = 60 * 60 * 8
        elif candle_type == '12hour':
            delta = 60 * 60 * 12
        elif candle_type == '1day':
            delta = 60 * 60 * 24
        elif candle_type == '1week':
            delta = 60 * 60 * 24 * 7
        elif candle_type == '1month':
            days_num = calendar.monthrange(current_dt.year, current_dt.month)[1]
            delta = 60 * 60 * 24 * days_num

        # current_dt = '2018-12-19 00:07:00' -> zaraba_time = '2018-12-19 00:05:00'
        zaraba_start = round_time(current_dt, date_delta=timedelta(seconds=delta), to='down')
        zaraba_end = round_time(current_dt, date_delta=timedelta(seconds=delta), to='up')
        if target_dt < zaraba_start:
            return 'completed'
        elif (zaraba_start <= target_dt) and (target_dt < zaraba_end):
            return 'working'
        else:
            return 'waiting'

    def get_candles(self, candle_type, start_dt, end_dt):
        start_dt = str2timestamp(start_dt)
        end_dt = str2timestamp(end_dt)
        now = self.tm.get_timestamp()
        assert (start_dt <= end_dt)
        assert (start_dt <= now)
        assert (end_dt <= now)

        # start_dt〜未確定の最新データ
        candle_status = self.get_candle_status(datetime.fromtimestamp(end_dt / 1000), datetime.fromtimestamp(now / 1000), candle_type)
        if candle_status == 'working':
            # 全データから最新データを取得
            dt = self.messages[candle_type].index
            mask = (start_dt <= dt) & (dt <= end_dt)
            latest_candle = self.messages[candle_type][mask].values.astype(np.float64)[-1].reshape(1, 6)
            # summaryデータから(最新-1)データまでを取得
            dt_summary = self.messages_summary[candle_type].index
            mask_summary = (start_dt <= dt_summary) & (dt_summary <= end_dt)
            candles_summary = self.messages_summary[candle_type][mask_summary].values.astype(np.float64)[:-1]
            candles = np.concatenate([candles_summary, latest_candle])
        elif candle_status == 'completed':
            dt_summary = self.messages_summary[candle_type].index
            mask_summary = (start_dt <= dt_summary) & (dt_summary <= end_dt)
            candles_summary = self.messages_summary[candle_type][mask_summary].values.astype(np.float64)
            candles = candles_summary
        else:
            print('error: candle_status == waiting')
            sys.exit()
        return candles