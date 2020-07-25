import os
import sys
import logging
import traceback
from time import sleep
from datetime import datetime, timedelta
from pubnub.pubnub import SubscribeListener

from trade.exception import InsufficientAssetsError, MarketOrderLimitError
from common.utils import candle_type_elem

root_module = os.path.basename(sys.argv[0])
if root_module == 'tr.py':
    logger = logging.getLogger('crypto')
elif root_module == 'sim.py':
    logger = logging.getLogger('simulate')
else:
    pass

class TGW(SubscribeListener):
    def __init__(self, candle_type, trader):
        super().__init__()
        self.candle_type = candle_type
        self.trader = trader
        self.time_limits = datetime.now()
        self.candle_stack = []

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
        if not self.valid_message(message):
            return

        curr_candle = message.message['data']['candlestick'][candle_type_elem(self.candle_type)]['ohlcv'][0]
        '''
        self.candle_stack.append(curr_candle)
        if len(self.candle_stack) == 1:
            return

        prev_candle = self.candle_stack[-2]
        if prev_candle[5] == curr_candle[5]:
            return        
        candle = prev_candle
        '''
        candle = curr_candle
        sleep(10)
        '''
        # 一定時間たったら成行に変更
        now = datetime.fromtimestamp(candle[5] / 1000)
        if (self.trader.order_type == 'limit') and (now > self.time_limits):
            self.trader.order_type = 'market'
        '''

        try:
            self.trader.trade(candle)
        except InsufficientAssetsError:
            logger.error(traceback.format_exc())
            logger.debug('process kill')
            sys.exit()
        except MarketOrderLimitError:
            logger.error(traceback.format_exc())
            logger.debug('change order type to limit')
            self.trader.order_type = 'limit'
            self.time_limits = datetime.fromtimestamp(candle[5] / 1000) + timedelta(minutes=5)
        except Exception:
            logger.error(traceback.format_exc())
            logger.debug('process reboot')
            os.execv(sys.executable, [sys.executable] + ['trade/tr.py'])