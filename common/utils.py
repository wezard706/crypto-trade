import numpy as np
import copy
import configparser
from datetime import timedelta, datetime
import pandas as pd
import seaborn as sn
import talib as ta
import matplotlib.pyplot as plt
from sklearn import metrics

inifile = configparser.ConfigParser()
inifile.read('config.ini', 'UTF-8')

def daterange(start, end):
    for n in range((end - start).days):
        yield start + timedelta(days=n)

def datetimerange(start, end):
    for n in range(int((end - start).total_seconds() / 60)):
        yield start + timedelta(minutes=n)

def round_time(self, dt=None, date_delta=timedelta(minutes=1), to='average'):
    """
    Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    dateDelta : timedelta object, we round to a multiple of this, default 1 minute.
    from:  http://.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object-python
    """
    round_to = date_delta.total_seconds()

    if dt is None:
        dt = datetime.now()
    seconds = (dt - dt.min).seconds

    if to == 'up':
        # // is a floor division, not a comment on following line (like in javascript):
        rounding = (seconds + round_to) // round_to * round_to
    elif to == 'down':
        rounding = seconds // round_to * round_to
    else:
        rounding = (seconds + round_to / 2) // round_to * round_to

    return dt + timedelta(0, rounding - seconds, -dt.microsecond)

def merge_list(list1, list2):
    list1_c = copy.deepcopy(list1)
    for l in list2:
        list1_c.append(l)
    return list1_c

def dict2str(d):
    message = ''
    for k, v in d.items():
        message += str(k) + ': ' + str(v) + ', '
    return message[:-1]

def extract_trades(log):
    trades = []
    for l in log:
        split_log = l.split(',')
        info_type = split_log[2]
        if info_type == ' [trade]':
            if len(split_log) != 9:
                continue
            log_time = split_log[0].replace('[INFO]', '')
            action = split_log[3].replace('action:', '').replace(' ', '')
            actual_price = float(split_log[4].replace('price:', ''))
            amount = float(split_log[5].replace('amount: ', ''))
            order_price = float(split_log[6].replace('order price: ', '').replace(' ', ''))
            trade_time = float(split_log[7].replace(' timestamp:', '').replace(' ', ''))
            trades.append([log_time, trade_time, action, amount, order_price, actual_price])
    return trades

def extract_params(log):
    params = []
    for l in log:
        split_log = l.split(',')
        info_type = split_log[2]
        if info_type == ' [params]':
            dt = split_log[0].replace('[INFO] ', '')
            pair = split_log[1].replace('pair: ', '')
            candle_type = split_log[2].replace('candle_type: ', '')
            timeperiod_s = split_log[3].replace('timeperiod_s: ', '')
            timeperiod_m = split_log[4].replace('timeperiod_m: ', '')
            timestep = split_log[5].replace('timestep: ', '')
            reward_rate = split_log[6].replace('reward_rate: ', '')
            risk_rate = split_log[7].replace('risk_rate: ', '')
            amount = split_log[8].replace('amount: ', '')
            asset_lowewr = split_log[9].replace('asset_lower: ', '')
            initial_assets = split_log[10]
            params.append([dt, pair, candle_type, timeperiod_s, timeperiod_m, timestep, reward_rate, risk_rate, amount,
                           asset_lowewr, initial_assets])
    return params

def calc_rocp_std(api, pair, start_dt, end_dt):
    # ボリンジャーバンド
    candles = api.get_candles(pair, candle_type='1min', start_dt=start_dt, end_dt=end_dt)
    candles = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
    candles.index = candles.timestamp.map(lambda x: datetime.fromtimestamp(x / 1000))
    bb_upper, bb_middle, bb_lower = ta.BBANDS(candles['close'], timeperiod=5, nbdevup=1, nbdevdn=1,
                                              matype=ta.MA_Type.EMA)

    c_std = bb_middle[-1] - bb_lower[-1]
    rocp_std = c_std / candles['close'][-1]
    return rocp_std

def plot_confusion_matrix(cm, out_path):
    plt.figure(figsize=(10, 7))
    sn.heatmap(cm, annot=True, cmap="Reds")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_path)

def calc_rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred)).astype('float32')

def str2timestamp(dt, format='%Y-%m-%d %H:%M:%S'):
    return datetime.strptime(dt, format).timestamp() * 1000

def str2dt(dt):
    return datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')

def dt2str(dt):
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def format_dt(dt, old_format, new_format):
    return datetime.strptime(dt, old_format).strftime(new_format)

def next_timestamp(timestamp, candle_type):
    if candle_type == '1min':
        return timestamp + 60000
    elif candle_type == '5min':
        return timestamp + 300000
    elif candle_type == '15min':
        return timestamp + 900000
    elif candle_type == '30min':
        return timestamp + 1800000
    elif candle_type == '1hour':
        return timestamp + 3600000
    elif candle_type == '4hour':
        return timestamp + 14400000
    elif candle_type == '8hour':
        return timestamp + 28800000
    elif candle_type == '12hour':
        return timestamp + 43200000
    elif candle_type == '1day':
        return timestamp + 86400000
    elif candle_type == '1week':
        return timestamp + 604800000
    elif candle_type == '1month':
        return

def calc_marubozu(open, close):
    body = close - open
    upper = np.mean(body) + np.std(body) * 1
    lower = np.mean(body) - np.std(body) * 1
    marubozu = np.zeros(len(open))
    marubozu[body > upper] = 100
    marubozu[body < lower] = -100
    return marubozu

def get_dt_format():
    return '%Y-%m-%d %H:%M:%S'

def candle_type_elem(candle_type):
    if candle_type == '1min':
        return 0
    elif candle_type == '5min':
        return 1
    elif candle_type == '15min':
        return 2
    elif candle_type == '30min':
        return 3
    elif candle_type == '1hour':
        return 4
    elif candle_type == '4hour':
        return 5
    elif candle_type == '8hour':
        return 6
    elif candle_type == '12hour':
        return 7
    elif candle_type == '1day':
        return 8
    elif candle_type == '1week':
        return 9
    elif candle_type == '1month':
        return 10


def get_api_key():
    return inifile.get('config', 'api_key')

def get_api_secret():
    return inifile.get('config', 'api_secret')

def get_subscribe_key():
    return inifile.get('config', 'subscribe_key')

def get_info_url():
    return inifile.get('slack', 'info_url')

def get_alert_url():
    return inifile.get('slack', 'alert_url')

def parse_error(e):
    message = e.args
    if (type(message) == tuple) and (type(message[0] == str)):
        if ('エラーコード' in message[0]) and ('内容' in message[0]):
            for code, _ in ERROR_CODES.items():
                if code == message[0][8:13]:
                    return code
    return '-1'

ERROR_CODES = {
    '10000': 'URLが存在しません',
    '10001': 'システムエラーが発生しました。サポートにお問い合わせ下さい',
    '10002': '不正なJSON形式です。送信内容をご確認下さい',
    '10003': 'システムエラーが発生しました。サポートにお問い合わせ下さい',
    '10005': 'タイムアウトエラーが発生しました。しばらく間をおいて再度実行して下さい',
    '20001': 'API認証に失敗しました',
    '20002': 'APIキーが不正です',
    '20003': 'APIキーが存在しません',
    '20004': 'API Nonceが存在しません',
    '20005': 'APIシグネチャが存在しません',
    '20011': '２段階認証に失敗しました',
    '20014': 'SMS認証に失敗しました',
    '30001': '注文数量を指定して下さい',
    '30006': '注文IDを指定して下さい',
    '30007': '注文ID配列を指定して下さい',
    '30009': '銘柄を指定して下さい',
    '30012': '注文価格を指定して下さい',
    '30013': '売買どちらかを指定して下さい',
    '30015': '注文タイプを指定して下さい',
    '30016': 'アセット名を指定して下さい',
    '30019': 'uuidを指定して下さい',
    '30039': '出金額を指定して下さい',
    '40001': '注文数量が不正です',
    '40006': 'count値が不正です',
    '40007': '終了時期が不正です',
    '40008': 'end_id値が不正です',
    '40009': 'from_id値が不正です',
    '40013': '注文IDが不正です',
    '40014': '注文ID配列が不正です',
    '40015': '指定された注文が多すぎます',
    '40017': '銘柄名が不正です',
    '40020': '注文価格が不正です',
    '40021': '売買区分が不正です',
    '40022': '開始時期が不正です',
    '40024': '注文タイプが不正です',
    '40025': 'アセット名が不正です',
    '40028': 'uuidが不正です',
    '40048': '出金額が不正です',
    '50003': '現在、このアカウントはご指定の操作を実行できない状態となっております。サポートにお問い合わせ下さい',
    '50004': '現在、このアカウントは仮登録の状態となっております。アカウント登録完了後、再度お試し下さい',
    '50005': '現在、このアカウントはロックされております。サポートにお問い合わせ下さい',
    '50006': '現在、このアカウントはロックされております。サポートにお問い合わせ下さい',
    '50008': 'ユーザの本人確認が完了していません',
    '50009': 'ご指定の注文は存在しません',
    '50010': 'ご指定の注文はキャンセルできません',
    '50011': 'APIが見つかりません',
    '60001': '保有数量が不足しています',
    '60002': '成行買い注文の数量上限を上回っています',
    '60003': '指定した数量が制限を超えています',
    '60004': '指定した数量がしきい値を下回っています',
    '60005': '指定した価格が上限を上回っています',
    '60006': '指定した価格が下限を下回っています',
    '70001': 'システムエラーが発生しました。サポートにお問い合わせ下さい',
    '70002': 'システムエラーが発生しました。サポートにお問い合わせ下さい',
    '70003': 'システムエラーが発生しました。サポートにお問い合わせ下さい',
    '70004': '現在取引停止中のため、注文を承ることができません',
    '70005': '現在買注文停止中のため、注文を承ることができません',
    '70006': '現在売注文停止中のため、注文を承ることができません',
    '70009': 'ただいま成行注文を一時的に制限しています。指値注文をご利用ください。',
    '70010': 'ただいまシステム負荷が高まっているため、最小注文数量を一時的に引き上げています。'
}