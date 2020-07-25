import os
import sys
import logging
import time
import threading
import traceback
from datetime import datetime, timedelta

from trade.exception import InsufficientAssetsError, APIAuthenticicationError

root_module = os.path.basename(sys.argv[0])
logger = logging.getLogger('crypto')

class TimeTradeEnvironment():
    def __init__(self, candle_type, trader, api, pair):
        super().__init__()
        self.candle_type = candle_type
        self.trader = trader
        self.api = api
        self.pair = pair

    def worker(self):
        start_dt = (datetime.now() - timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')
        end_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        candle = self.api.get_candles(self.pair, candle_type=self.candle_type, start_dt=start_dt, end_dt=end_dt)[0]
        logger.debug('get candle: {} {} {} {} {}'.format(candle[0], candle[1], candle[2], candle[3], candle[4]))

        try:
            self.trader.trade(candle)
        except InsufficientAssetsError:
            logger.error(traceback.format_exc())
            logger.debug('process kill')
            sys.exit()
        except APIAuthenticicationError:
            logger.error(traceback.format_exc())
        except Exception:
            logger.error(traceback.format_exc())
            logger.debug('process reboot')
            os.execv(sys.executable, [sys.executable] + ['trade/tr.py'])

    def schedule(self, interval, f, wait=True):
        base_time = time.time()
        next_time = 0
        while True:
            t = threading.Thread(target=f)
            t.start()
            if wait:
                t.join()
            next_time = ((base_time - time.time()) % interval) or interval
            time.sleep(next_time)

    def run(self):
        # 一定時刻間隔でトレードを実施する
        while True:
            dt = datetime.now()
            # 現在時刻が00[s]になるまで待機
            if str(dt.second) == "0":
                self.schedule(interval=10, f=self.worker, wait=False)