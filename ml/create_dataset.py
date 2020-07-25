import time
import pandas as pd
from datetime import datetime, timedelta
import mysql.connector
import configparser
from urllib.parse import urlparse
from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub

from trade.api import API
from common.utils import get_subscribe_key

# DB設定
inifile = configparser.ConfigParser()
inifile.read('config.ini', 'UTF-8')
user = inifile.get('mysql', 'user')
password = inifile.get('mysql', 'password')
alert_url = inifile.get('slack', 'alert_url')

# PubNubの設定
pnconfig = PNConfiguration()
pnconfig.subscribe_key = get_subscribe_key()
pnconfig.ssl = True
pubnub = PubNub(pnconfig)


def format_candles(candles_all):
    format_candles = {}
    candle_types = ['1min', '5min', '15min', '30min', '1hour', '4hour', '8hour', '12hour', '1day', '1week', '1month']
    for ctype in candle_types:
        candles = candles_all.loc[:, [s + '_' + ctype for s in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]]
        candles.columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        format_candles[ctype] = candles.groupby(['timestamp']).tail(1)
    return format_candles

def create_dataset_db(start_dt, end_dt):
    url = urlparse('mysql://' + user + ':' + password + '@localhost:3306/crypto')
    conn = mysql.connector.connect(
        host=url.hostname or 'localhost',
        port=url.port or 3306,
        user=url.username or 'root',
        password=url.password or '',
        database=url.path[1:],
    )
    start_timestamp = datetime.strptime(start_dt, '%Y-%m-%d %H:%M:%S').timestamp() * 1000
    end_timestamp = datetime.strptime(end_dt, '%Y-%m-%d %H:%M:%S').timestamp() * 1000

    # クエリを実行
    cur = conn.cursor(dictionary=True)
    cur.execute('SELECT * FROM candlestick WHERE %s <= timestamp AND timestamp <= %s', [start_timestamp, end_timestamp])
    candles_all = pd.DataFrame(cur.fetchall())
    candles_all.to_pickle('ml/input/201901191040_201901201039_candles.pkl')
    #candles_all = format_candles(candles_all)
    return candles_all

def create_dataset_api(start_dt, end_dt, pair, candle_type):
    start_dt = datetime.strptime(start_dt, '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(end_dt, '%Y-%m-%d %H:%M:%S')
    candles_all = None
    while True:
        end_dt_split = start_dt + timedelta(days=7) - timedelta(seconds=1)
        if end_dt_split < end_dt:
            print('get　candle during {} - {}'.format(start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt_split.strftime('%Y-%m-%d %H:%M:%S')))
            candles = API().get_candles(pair, candle_type=candle_type,
                                        start_dt=start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                                        end_dt=end_dt_split.strftime('%Y-%m-%d %H:%M:%S'))
            start_dt = end_dt_split + timedelta(seconds=1)
            candles = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
            if candles_all is None:
                candles_all = pd.DataFrame(candles)
            else:
                candles_all = pd.concat([candles_all, candles])
        else:
            print('get candle during {} - {}'.format(start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S')))
            candles = API().get_candles(pair, candle_type=candle_type,
                                        start_dt=start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                                        end_dt=end_dt.strftime('%Y-%m-%d %H:%M:%S'))
            candles = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
            if candles_all is None:
                candles_all = pd.DataFrame(candles)
            else:
                candles_all = pd.concat([candles_all, candles])
            break
        time.sleep(10)

    return candles_all

if __name__=='__main__':
    #create_dataset_db(start_dt='2019-01-19 10:40:00', end_dt='2019-01-20 10:39:59')
    candle_types = ['1min', '5min']
    for candle_type in candle_types:
        print(candle_type)
        create_dataset_api(start_dt='2018-11-30 20:55:00', end_dt='2019-01-23 23:59:59', pair='xrp_jpy', candle_type=candle_type)
