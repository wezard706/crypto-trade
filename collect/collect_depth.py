import os
import sys
import traceback
import logging
import time
import threading
import python_bitbankcc
import redis
import json

from common.utils import get_api_key, get_api_secret, get_info_url
from trade.api import API

from datetime import datetime, timedelta

prv = python_bitbankcc.private(get_api_key(), get_api_secret())
pub = python_bitbankcc.public()

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

from common.utils import str2dt, dt2str, format_dt

def save_depth(pair='xrp_jpy', db=0):
    pool = redis.ConnectionPool(host='localhost', port=6379, db=db)
    conn = redis.StrictRedis(connection_pool=pool)

    depth = API().get_depth(pair)
    dt = str2dt(format_dt(dt2str(datetime.now()), '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S'))
    conn.set(dt2str(dt), json.dumps(depth))
    logger.debug('save depth at timestamp: {}'.format(dt))

def save_transactions(pair='xrp_jpy', db=1):
    pool = redis.ConnectionPool(host='localhost', port=6379, db=db)
    conn = redis.StrictRedis(connection_pool=pool)

    end_dt = str2dt(format_dt(dt2str(datetime.now()), '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S'))
    start_dt = end_dt - timedelta(seconds=5)
    #logger.debug('get transactions from {} to {}'.format(start_dt, end_dt))
    transactions = API().get_transactions(pair, start_dt, end_dt)
    conn.set(dt2str(end_dt), json.dumps(transactions))
    logger.debug('save transactions at timestamp: {}'.format(end_dt))

def worker():
    try:
        save_depth(pair='xrp_jpy', db=0)
        save_transactions(pair='xrp_jpy', db=1)
    except Exception:
        logger.error(traceback.format_exc())
        logger.debug('process reboot')
        os.execv(sys.executable, [sys.executable] + ['collect/collect_depth.py'])

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

def collect_depth():
    # 一定時刻間隔でトレードを実施する
    while True:
        dt = datetime.now()
        # 現在時刻が00[s]になるまで待機
        if str(dt.second) == "0":
            logger.debug('start schduler: {}'.format(dt))
            # スケジューラを起動する（intervalごとにトレードを実施する）
            schedule(interval=10, f=worker, wait=False)

if __name__=='__main__':
    collect_depth()