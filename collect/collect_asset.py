import os
import sys
import traceback
import logging
import mysql.connector
import configparser
from urllib.parse import urlparse
from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub
from slack_log_handler import SlackLogHandler
from datetime import datetime
import time
import threading
import python_bitbankcc

from common.utils import get_subscribe_key
from pubnub.pubnub import SubscribeListener

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

# logger
logger = logging.getLogger('collect')
logger.setLevel(logging.DEBUG)
format = logging.Formatter('[%(levelname)s] %(asctime)s, %(message)s')
# 標準出力
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(format)
logger.addHandler(stream_handler)
# ファイル出力(debug)
dfile_handler = logging.FileHandler('collect/log/debug.log', 'a')
dfile_handler.setLevel(logging.DEBUG)
dfile_handler.setFormatter(format)
logger.addHandler(dfile_handler)
# ファイル出力(error)
efile_handler = logging.FileHandler('collect/log/error.log', 'a')
efile_handler.setLevel(logging.ERROR)
efile_handler.setFormatter(format)
logger.addHandler(efile_handler)
# slack
slack_handler = SlackLogHandler(
    webhook_url=alert_url,
    emojis={
        logging.INFO: ':grinning:',
        logging.WARNING: ':white_frowning_face:',
        logging.ERROR: ':persevere:',
        logging.CRITICAL: ':confounded:',
    }
)
slack_handler.setLevel(logging.ERROR)
slack_handler.setFormatter(format)
logger.addHandler(slack_handler)

class SimpleListener(SubscribeListener):
    def __init__(self, conn):
        super().__init__()
        self.conn = conn

    def create_query(self):
        query = 'INSERT INTO crypto.candlestick (' \
                'pid, channel, timetoken, timestamp, ' \
                'open_1min, high_1min, low_1min, close_1min, volume_1min, timestamp_1min,' \
                'open_5min, high_5min, low_5min, close_5min, volume_5min, timestamp_5min,' \
                'open_15min, high_15min, low_15min, close_15min, volume_15min, timestamp_15min,' \
                'open_30min, high_30min, low_30min, close_30min, volume_30min, timestamp_30min,' \
                'open_1hour, high_1hour, low_1hour, close_1hour, volume_1hour, timestamp_1hour,' \
                'open_4hour, high_4hour, low_4hour, close_4hour, volume_4hour, timestamp_4hour,' \
                'open_8hour, high_8hour, low_8hour, close_8hour, volume_8hour, timestamp_8hour,' \
                'open_12hour, high_12hour, low_12hour, close_12hour, volume_12hour, timestamp_12hour,' \
                'open_1day, high_1day, low_1day, close_1day, volume_1day, timestamp_1day,' \
                'open_1week, high_1week, low_1week, close_1week, volume_1week, timestamp_1week,' \
                'open_1month, high_1month, low_1month, close_1month, volume_1month, timestamp_1month)' \
                'values (' \
                '%s, %s, %s, %s,' \
                '%s, %s, %s, %s, %s, %s,' \
                '%s, %s, %s, %s, %s, %s,' \
                '%s, %s, %s, %s, %s, %s,' \
                '%s, %s, %s, %s, %s, %s,' \
                '%s, %s, %s, %s, %s, %s,' \
                '%s, %s, %s, %s, %s, %s,' \
                '%s, %s, %s, %s, %s, %s,' \
                '%s, %s, %s, %s, %s, %s,' \
                '%s, %s, %s, %s, %s, %s,' \
                '%s, %s, %s, %s, %s, %s,' \
                '%s, %s, %s, %s, %s, %s);'
        return query

    def valid_message(self, message):
        candles = message.message['data']['candlestick']
        if None in candles:
            logger.debug("None in candle")
            return False
        elif len(candles) != 11:
            logger.debug("len(candle) != 11")
            return False
        else:
            return True

    def message(self, pubnub, message):
        logger.debug(message.message)
        if not self.valid_message(message):
            return

        # format insert data
        pid = str(message.message['pid'])
        channel = message.channel
        timetoken = str(message.timetoken)
        timestamp = str(message.message['data']['timestamp'])
        insert_data = [pid, channel, timetoken, timestamp]

        candles = message.message['data']['candlestick']
        for c in candles:
            for ohlcv in c['ohlcv'][0]:
                insert_data.append(ohlcv)

        # クエリを実行
        query = self.create_query()
        cur = self.conn.cursor()
        try:
            cur.execute(query, insert_data)
            self.conn.commit()
        except:
            self.conn.rollback()
            logger.error(traceback.format_exc())
            logger.debug('process reboot')
            os.execv(sys.executable, [sys.executable] + ['collect/collect_asset.py'])
        cur.close()

def worker():
    try:
        assets = prv.get_asset()
        asset_xrp = assets['assets'][3]['onhand_amount']
        asset_jpy = assets['assets'][0]['onhand_amount']

        start_dt = (datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
        end_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        from trade.api import API
        pair = 'xrp_jpy'
        candle_type = '1min'
        candles = API().get_candles(pair, candle_type, start_dt, end_dt)
        latest_close = candles[-1][3]
        asset_all = float(asset_jpy) + float(asset_xrp) * latest_close
        logger.debug('jpy: {}, xrp: {}, all: {}'.format(asset_jpy, asset_xrp, asset_all))

    except Exception:
        logger.error(traceback.format_exc())
        logger.debug('process reboot')
        os.execv(sys.executable, [sys.executable] + ['collect/collect_asset.py'])

def schedule(interval, f, wait=True):
    base_time = time.time()
    next_time = 0
    while True:
        t = threading.Thread(target=f)
        t.start()
        if wait:
            t.join()
        next_time = ((base_time - time.time()) % interval) or interval
        time.sleep(next_time)

def collect_asset():
    # 一定時刻間隔でトレードを実施する
    while True:
        dt = datetime.now()
        # 現在時刻が00[s]になるまで待機
        if str(dt.second) == "0":
            logger.debug('start schduler: {}'.format(dt))
            # スケジューラを起動する（intervalごとにトレードを実施する）
            schedule(interval=60, f=worker, wait=False)

if __name__=='__main__':
    collect_asset()