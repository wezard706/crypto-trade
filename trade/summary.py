import matplotlib.pyplot as plt
import sys
import numpy as np
import python_bitbankcc
import slackweb

from common.utils import get_api_key, get_api_secret, get_info_url
from common.chart_creator import ChartCreator as cc

import pandas as pd
import configparser
from urllib.parse import urlparse
from datetime import datetime, timedelta
import mysql.connector

# DB設定
inifile = configparser.ConfigParser()
inifile.read('config.ini', 'UTF-8')
user = inifile.get('mysql', 'user')
password = inifile.get('mysql', 'password')
alert_url = inifile.get('slack', 'alert_url')

prv = python_bitbankcc.private(get_api_key(), get_api_secret())
slack_info = slackweb.Slack(url=get_info_url())

def check_buysell_order(df):
    for i in range(0, len(df), 2):
        if not (df.iloc[i].side == 'buy' and df.iloc[i + 1].side == 'sell'):
            return False
    return True

def format_trade_history(history, start_dt, end_dt):
    history = history.astype({'amount': float, 'executed_at': int, 'fee_amount_base': float, 'fee_amount_quote': float,
                              'maker_taker': str, 'order_id': int, 'pair': str, 'price': float, 'side': str,
                              'trade_id': int, 'type': str})
    history.index = history.executed_at.apply(lambda x: datetime.fromtimestamp(x / 1000))
    history = history[(history.index >= start_dt) & (history.index <= end_dt)]

    # 先頭は’buy’からカウント
    for k, v in history.iterrows():
        if v.side == 'buy':
            break
        history.drop(index=k, inplace=True)
        print('delete row with index {}'.format(k))

    # 末尾は’sell’まで
    for k, v in history.sort_index(ascending=False).iterrows():
        if v.side == 'sell':
            break
        history.drop(index=k, inplace=True)
        print('delete row with index {}'.format(k))

    return history


def drop_strange_history(history, amount):
    '''
    - 同じorder_idはまとめる
    - 購入量がamountに満たない履歴は削除
    '''

    grouped_history = history.groupby('order_id').agg({'amount': sum, 'executed_at': 'last', 'fee_amount_base': 'last',
                                               'fee_amount_quote': 'last', 'maker_taker': 'last', 'order_id': 'last',
                                               'pair': 'last', 'price': np.mean, 'side': 'last', 'trade_id': 'last',
                                               'type': 'last'})
    # 購入量が10より小さいものは削除（約定途中？）
    grouped_history = grouped_history[grouped_history.amount == amount]
    grouped_history.index = grouped_history.executed_at.apply(lambda x: datetime.fromtimestamp(x / 1000))
    return grouped_history

def format_message(messages):
    candles_all = {}
    candle_types = ['1min', '5min', '15min', '30min', '1hour', '4hour', '8hour', '12hour', '1day', '1week', '1month']
    for ctype in candle_types:
        candles = messages.loc[:, [s + '_' + ctype for s in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]]
        candles.columns = ['open', 'high', 'low', 'close', 'volume', 'unixtime']
        candles['unixtime'] = candles['unixtime'] / 1000
        candles_all[ctype] = candles
    return candles_all

def summary_candle(df):
    return df.groupby('unixtime').agg({'open': lambda x: x.iloc[0], 'high': max, 'low': min, 'close': lambda x: x.iloc[-1], 'volume': lambda x: x.iloc[-1], 'unixtime': lambda x: x.iloc[-1]})

def summary(start_dt, end_dt, candle_type):
    start_dt = datetime.strptime(start_dt, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(end_dt, '%Y-%m-%d %H:%M:%S')

    history = pd.DataFrame(prv.get_trade_history(pair='xrp_jpy', order_count=1000)['trades']).sort_index(ascending=False)
    formatted_history = format_trade_history(history, start_dt, end_dt)

    # 期間全体の利益を計算
    history_buy = formatted_history[formatted_history.side == 'buy']
    history_sell = formatted_history[formatted_history.side == 'sell']
    price_buy = history_buy.price * history_buy.amount
    price_sell = history_sell.price * history_sell.amount
    total_profit = price_sell.sum() - price_buy.sum()
    total_amount_buy = history_buy.amount.sum()
    total_amount_sell = history_sell.amount.sum()
    if total_amount_buy > total_amount_sell:
        rest = total_amount_buy - total_amount_sell
        print('利益: {:.4f}, 余剰(buy): {:.4f}'.format(total_profit, rest))
    elif total_amount_buy < total_amount_sell:
        rest = total_amount_sell - total_amount_buy
        print('利益: {:.4f}, 余剰(sell): {:.4f}'.format(total_profit, rest))
    else:
        print('利益: {:.4f}, 余剰: 0'.format(total_profit))

    assets = []
    idx_buy = 0
    idx_sell = 0
    price_buy = 0
    amount_buy = 0
    history_buy['executed_at_ms'] = pd.to_datetime(history_buy.index.map(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S')))
    history_sell['executed_at_ms'] = pd.to_datetime(history_sell.index.map(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S')))
    history_buy = history_buy.groupby(['executed_at_ms']).agg({'price': np.mean, 'amount': sum})
    history_sell = history_sell.groupby(['executed_at_ms']).agg({'price': np.mean, 'amount': sum})
    for dt in pd.date_range(start_dt, end_dt, freq='S'):
        profit = 0
        if (len(history_buy) > idx_buy) and (history_buy.index[idx_buy] == dt):
            price_buy += history_buy.ix[idx_buy, 'price'] * history_buy.ix[idx_buy, 'amount']
            amount_buy += history_buy.ix[idx_buy, 'amount']
            idx_buy += 1
            print('buy {} {} {} {}'.format(idx_buy, dt, price_buy, profit))
        elif (len(history_sell) > idx_sell) and (history_sell.index[idx_sell] == dt):
            if price_buy > 0:
                #price_sell = history_sell.ix[idx_sell, 'price'] * history_sell.ix[idx_sell, 'amount']
                price_sell = history_sell.ix[idx_sell, 'price'] * amount_buy
                profit = price_sell - price_buy
                print('{} {} = {} - {}'.format(dt, profit, price_sell, price_buy))
                #print('profit {} {}'.format(dt, profit))
                price_buy = 0
                amount_buy = 0
                idx_sell += 1

        if len(assets) > 0:
            assets.append(assets[-1] + profit)
        else:
            assets.append(profit)


if __name__=='__main__':
    start_dt = datetime.now().replace(hour=2, minute=24, second=0, microsecond=0)
    end_dt = (start_dt + timedelta(1)).strftime('%Y-%m-%d %H:%M:%S')
    start_dt = start_dt.strftime('%Y-%m-%d %H:%M:%S')
    start_dt = '2019-02-04 00:00:00'
    end_dt = '2019-02-04 07:59:59'
    candle_type = '1min'
    summary(start_dt, end_dt, candle_type)

