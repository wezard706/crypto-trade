import os
import sys
import logging
import traceback
from pubnub.models.consumer.pubsub import PNMessageResult
from datetime import datetime, timedelta

from trade.exception import InsufficientAssetsError, MarketOrderLimitError
from common.utils import candle_type_elem

root_module = os.path.basename(sys.argv[0])
if root_module == 'tr.py':
    logger = logging.getLogger('crypto')
elif root_module == 'sim.py':
    logger = logging.getLogger('simulate')
else:
    pass

class PubSubSimulator:
    def __init__(self, candle_type, trader, api, tm):
        self.candle_type = candle_type
        self.trader = trader
        self.api = api
        self.tm = tm
        self.candle_stack = []

    def publish(self, messages):
        message_size = len(messages['1min'])
        for i in range(message_size):                        
            data = {
                'pid': i,
                'data': {
                    'candlestick': [
                        {'type': '1min', 'ohlcv': [messages['1min'].iloc[i]]},
                        {'type': '5min', 'ohlcv': [messages['5min'].iloc[i]]},
                        {'type': '15min', 'ohlcv': [messages['15min'].iloc[i]]},
                        {'type': '30min', 'ohlcv': [messages['30min'].iloc[i]]},
                        {'type': '1hour', 'ohlcv': [messages['1hour'].iloc[i]]},
                        {'type': '4hour', 'ohlcv': [messages['4hour'].iloc[i]]},
                        {'type': '8hour', 'ohlcv': [messages['8hour'].iloc[i]]},
                        {'type': '12hour', 'ohlcv': [messages['12hour'].iloc[i]]},
                        {'type': '1day', 'ohlcv': [messages['1day'].iloc[i]]},
                        {'type': '1week', 'ohlcv': [messages['1week'].iloc[i]]},
                        {'type': '1month', 'ohlcv': [messages['1month'].iloc[i]]}
                    ],
                    'timestamp': [messages['1min'].index[i]]
                }
            }

            pnmr = PNMessageResult(data, subscription='', channel='simulation', timetoken=i)
            yield pnmr

    def subscribe(self, message):
        if not self.valid_message(message):
            return

        self.candle_stack.append(message.message['data']['candlestick'][candle_type_elem(self.candle_type)]['ohlcv'][0])
        if len(self.candle_stack) == 1:
            return

        prev_candle = self.candle_stack[-2]
        latest_candle = self.candle_stack[-1]
        if prev_candle[5] == latest_candle[5]:
            return
        candle = prev_candle

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
            os.execv(sys.executable, [sys.executable] + ['simulate/sim.py'])

    def run(self, messages):
        for i, message in enumerate(self.publish(messages)):
            self.tm.update_timestamp()
            if i % 100 == 0:
                logger.debug('{} / {}'.format(i, len(messages['1min'])))
            self.subscribe(message)

    def valid_message(self, message):
        candles = message.message['data']['candlestick']
        if None in candles:
            print("None in candle")
            return False
        elif len(candles) != 11:
            print("len(candle) != 11")
            return False
        else:
            return True
