import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta

from trade.api import API
import python_bitbankcc
from datetime import datetime, timedelta

from common.utils import get_api_key, get_api_secret, calc_rocp_std
from ml.train_regression import _extract_feature, get_params
prv = python_bitbankcc.private(get_api_key(), get_api_secret())

def find_contiguous_colors(colors):
    # finds the continuous segments of colors and returns those segments
    segs = []
    curr_seg = []
    prev_color = ''
    for c in colors:
        if c == prev_color or prev_color == '':
            curr_seg.append(c)
        else:
            segs.append(curr_seg)
            curr_seg = []
            curr_seg.append(c)
        prev_color = c
    segs.append(curr_seg)  # the final one
    return segs

def plot_multicolored_lines(x, y, colors):
    fig, ax = plt.subplots(2, 1, figsize=(18, 10))

    segments = find_contiguous_colors(colors)
    start = 0
    plt.hold(True)
    for i, seg in enumerate(segments):
        end = start + len(seg) + 1
        l, = ax[0].plot(x[start:end], y[start:end], c=seg[0], lw=0.75)
        start = end - 1
    return ax

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

def get_candles_for_feature(candle_types, start_dt, end_dt):
    candles_for_feature = {}
    for candle_type in candle_types:
        candles = API().get_candles(pair, candle_type=candle_type,
                                start_dt=(start_dt - timedelta(1)).strftime('%Y-%m-%d %H:%M:%S'),
                                end_dt=end_dt.strftime('%Y-%m-%d %H:%M:%S'))
        candles = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
        candles.index = candles.timestamp.map(lambda x: datetime.fromtimestamp(x / 1000))
        candles_for_feature[candle_type] = candles
    return candles_for_feature

def main(pair, candle_type, start_dt, end_dt, is_predict):
    # 買い〜売り：RED
    # 売り〜買い：BLACK
    '''
    start_dt = datetime.strptime(start_dt, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(end_dt, '%Y-%m-%d %H:%M:%S')

    # トレード履歴を取得してフォーマット
    with open(dirpath) as f:
        debug_log = [line for line in f.readlines() if '[INFO]' in line]
    trades = extract_trades(debug_log)
    trades = pd.DataFrame(trades)
    trades.columns = ['log_time', 'trade_time', 'action', 'amount', 'order_price', 'actual_price']
    trades['trade_time'] = pd.to_datetime(trades['trade_time'].apply(lambda x: datetime.fromtimestamp(x / 1000)))
    trades = trades[(trades['trade_time'] >= start_dt) & (trades['trade_time'] < end_dt)]

    # 先頭は’buy’からカウント
    if (trades.action.iloc[0] == 'exit_long') or (trades.action.iloc[0] == 'exit_short'):
        trades = trades[1:]
    # 末尾は'sell'まで
    if (trades.action.iloc[trades.shape[0] - 1] == 'entry_long') or (trades.action.iloc[trades.shape[0] - 1] == 'entry_short'):
        trades = trades[:-1]

    # 指定した時刻のローソク足を取得
    candles = API().get_candles(pair, candle_type, start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S'))
    x = pd.to_datetime([datetime.fromtimestamp(c[5] / 1000).strftime('%Y-%m-%d %H:%M:%S') for c in candles])
    y = [float(c[3]) for c in candles]
    '''

    start_dt = datetime.strptime(start_dt, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(end_dt, '%Y-%m-%d %H:%M:%S')

    history = pd.DataFrame(prv.get_trade_history(pair='xrp_jpy', order_count=10000)['trades']).sort_index(ascending=False)
    formatted_history = format_trade_history(history, start_dt, end_dt)
    clean_history = drop_strange_history(formatted_history, amount=10)
    if not check_buysell_order(clean_history):
        print('order of trade type is invalid')
        sys.exit()

    # 指定した時刻のローソク足を取得
    candles = API().get_candles(pair, candle_type, start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S'))
    x = pd.to_datetime([datetime.fromtimestamp(c[5] / 1000).strftime('%Y-%m-%d %H:%M:%S') for c in candles])
    y = [float(c[3]) for c in candles]

    # トレード履歴からカラーマップを作成
    colors = pd.Series(['BLACK'] * len(x), index=x)
    prev_trade_dt = start_dt
    for curr_trade_dt, row in clean_history.iterrows():
        curr_trade_dt = curr_trade_dt.strftime('%Y-%m-%d %H:%M:00')
        if row['side'] == 'buy':
            colors[(colors.index >= prev_trade_dt) & (colors.index <= curr_trade_dt)] = 'BLACK'
        elif row['side'] == 'sell':
            colors[(colors.index >= prev_trade_dt) & (colors.index <= curr_trade_dt)] = 'RED'
        prev_trade_dt = curr_trade_dt

    ax = plot_multicolored_lines(x, y, colors)

    if is_predict:
        with open('ml/model/clf_lower.pkl', mode='rb') as f:
            clf_lower = pickle.load(f)

        candle_types = ['1min', '5min', '15min', '30min']
        candles_for_feature = get_candles_for_feature(candle_types, start_dt, end_dt)

        results = []
        for candle in candles:
            end_dt = datetime.fromtimestamp(candle[5] / 1000)
            start_dt = (end_dt - timedelta(1)).strftime('%Y-%m-%d %H:%M:%S')
            end_dt = end_dt.strftime('%Y-%m-%d %H:%M:%S')
            print(end_dt)

            features = None
            for candle_type in candle_types:
                #candles = API().get_candles(pair, candle_type=candle_type, start_dt=start_dt, end_dt=end_dt)
                #candles = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
                #candles.index = candles.timestamp.map(lambda x: datetime.fromtimestamp(x / 1000))
                cff = candles_for_feature[candle_type]
                cff = cff[(datetime.strptime(start_dt, '%Y-%m-%d %H:%M:%S') <= cff.index) & (cff.index <= datetime.strptime(end_dt, '%Y-%m-%d %H:%M:%S'))]

                params = get_params('1min')
                feature = _extract_feature(cff, params)
                feature = feature.fillna(method='ffill')
                feature.columns = [c + '_' + candle_type for c in feature.columns]

                if features is None:
                    features = feature
                else:
                    features = pd.concat([features, feature], axis=1)
                    features = features.fillna(method='ffill')
            from time import sleep
            #sleep(3)
            # 予測値を低く見積もって（25%点）さらにROCPの分散も考慮した値が０より大きければロング
            X = features[features.index == end_dt]
            y_lower = clf_lower.predict(X)[0]
            rocp_std = calc_rocp_std(start_dt, end_dt)
            rocp_lower = y_lower - rocp_std
            results.append({'y_lower': y_lower, 'rocp_std': rocp_std, 'rocp_lower': rocp_lower})

        results = pd.DataFrame(results)
        with open('results.pkl', 'wb') as f:
            pickle.dump(results, f)
    else:
        with open('results.pkl', 'rb') as f:
            results = pickle.load(f)

    ax[1].plot(x, results.y_lower)
    ax[1].plot(x, results.rocp_std)
    ax[1].plot(x, results.rocp_lower)
    ax[1].legend(['y_lower', 'rocp_std', 'rocp_lower'])
    plt.show()

if __name__=='__main__':
    pair = 'xrp_jpy'
    candle_type = '1min'
    start_dt = '2019-01-12 00:00:00'
    end_dt = '2019-01-12 23:59:59'
    dirpath = 'trade/log/debug.log'
    main(pair, candle_type, start_dt, end_dt, is_predict=False)
