# logger
import logging
logger = logging.getLogger('simulate')
logger.setLevel(logging.DEBUG)
format = logging.Formatter('[%(levelname)s] %(asctime)s, %(message)s')
# 標準出力
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(format)
logger.addHandler(stream_handler)
# ファイル出力(debug)
dfile_handler = logging.FileHandler('simulate/log/debug.log', 'w')
dfile_handler.setLevel(logging.DEBUG)
dfile_handler.setFormatter(format)
logger.addHandler(dfile_handler)
# ファイル出力(error)
efile_handler = logging.FileHandler('simulate/log/error.log', 'w')
efile_handler.setLevel(logging.ERROR)
efile_handler.setFormatter(format)
logger.addHandler(efile_handler)

import pandas as pd
import configparser
from urllib.parse import urlparse
from datetime import datetime, timedelta
import mysql.connector

from trade.params import get_params
from trade.trader import Trader
from trade.model.modellgb import ModelLGB
from simulate.tm import TimeManager
from simulate.apisim import APISimulator
from simulate.rm import ResourceManager
from simulate.pubsubsim import PubSubSimulator
from common.utils import dict2str
from trade.summary import summary

# DB設定
inifile = configparser.ConfigParser()
inifile.read('config.ini', 'UTF-8')
user = inifile.get('mysql', 'user')
password = inifile.get('mysql', 'password')
alert_url = inifile.get('slack', 'alert_url')

def create_message(start_dt, end_dt):
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
    messages = pd.DataFrame(cur.fetchall())
    messages = format_message(messages)
    return messages

def format_message(messages):
    candles_all = {}
    candle_types = ['1min', '5min', '15min', '30min', '1hour', '4hour', '8hour', '12hour', '1day', '1week', '1month']
    for ctype in candle_types:
        candles = messages.loc[:, [s + '_' + ctype for s in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]]
        candles.columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        candles.index = messages['timestamp']
        candles_all[ctype] = candles
    return candles_all

def create_tickers(messages, spread=0.062):
    tickers = []
    for timestamp, message in messages['1min'].iterrows():
        sell = float(message['close'])
        buy = round(((100 - spread) / (100 + spread)) * sell, 3)
        tickers.append({'buy': buy, 'sell': sell})
    tickers = pd.DataFrame(tickers, index=messages['1min'].index)
    return tickers

def simulate():
    start_dt = '2018-11-20 00:00:00'
    end_dt = '2018-12-31 23:59:59'
    #end_dt = '2018-12-28 23:59:59'
    assets = 3000
    params = get_params()

    # 過去の値を含めたローソク足を作成
    start_dt_ext = (datetime.strptime(start_dt, '%Y-%m-%d %H:%M:%S') - timedelta(1)).strftime('%Y-%m-%d %H:%M:%S')
    messages_ext = create_message(start_dt_ext, end_dt)

    messages = create_message(start_dt, end_dt)
    tickers = create_tickers(messages)
    tm = TimeManager(timestamps=messages['1min'].index)
    rm = ResourceManager(messages_ext, tickers, assets, tm)  # ResourceMangerは最終的なローソク足のみ登録する
    api = APISimulator(rm)

    logger.info('[params], ' + dict2str(params) + 'initial_assets: ' + str(assets))
    #seller = ModelNaive(params['reward_upper'], params['loss_lower'], params['update_threshold'], params['step'])
    #buyer = ModelMa(params['pair'], params['candle_type'], params['timeperiod_s'], params['timeperiod_m'], params['timeperiod_l'], params['timeperiod_xl'], api)
    buyer = ModelLGB(params['pair'], api)
    seller = ModelLGB(params['pair'], api)
    trader = Trader(params['pair'], params['candle_type'], buyer, seller, params['amount'], params['order_type'], params['asset_lower'], api)
    PubSubSimulator(params['candle_type'], trader, api, tm).run(messages)
    summary(start_dt, end_dt, 'simulate/log/debug.log')

if __name__=='__main__':
    simulate()