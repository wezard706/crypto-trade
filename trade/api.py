import os
import sys
import time
import logging
import python_bitbankcc
import configparser
from abc import abstractmethod
from datetime import datetime, timedelta
from trade.exception import MarketOrderLimitError, CancelFailedError, APIAuthenticicationError
import numpy as np

from common.utils import merge_list, daterange, parse_error

root_module = os.path.basename(sys.argv[0])
if root_module == 'tr.py':
    logger = logging.getLogger('crypto')
elif root_module == 'sim.py':
    logger = logging.getLogger('simulate')
else:
    pass

inifile = configparser.ConfigParser()
inifile.read('config.ini', 'UTF-8')
api_key = inifile.get('config', 'api_key')
api_secret = inifile.get('config', 'api_secret')
prv = python_bitbankcc.private(api_key, api_secret)
pub = python_bitbankcc.public()

class APIBase():
    @abstractmethod
    def get_asset(self):
        pass

    @abstractmethod
    def get_ticker(self, pair):
        pass

    @abstractmethod
    def get_candle(self, pair, candle_type):
        pass

    @abstractmethod
    def get_candles(self, pair, candle_type, start_dt, end_dt):
        pass

    @abstractmethod
    def get_active_orders(self, pair, options=None):
        pass

    @abstractmethod
    def get_trade_history(self, pair, order_count):
        pass

    @abstractmethod
    def get_order(self, pair, order_id):
        pass

    @abstractmethod
    def order(self, pair, price, amount, side, order_type):
        pass

    @abstractmethod
    def cancel_order(self, pair, order_id):
        pass

    @abstractmethod
    def cancel_orders(self, pair, order_ids):
        pass

    @abstractmethod
    def get_transactions(self, pair, start_dt=None, end_dt=None):
        pass

    @abstractmethod
    def get_depth(self, pair):
        pass

class API(APIBase):
    def get_asset(self):
        try:
            return float(prv.get_asset()['assets'][0]['onhand_amount'])
        except Exception as e:
            if parse_error(e) == '20001':
                raise APIAuthenticicationError
            else:
                raise Exception

    def get_ticker(self, pair):
        return pub.get_ticker(pair)

    def get_candles(self, pair, candle_type, start_dt, end_dt):
        def _get_candles(pair, candle_type, start_date, end_date):
            '''
            pubhubのget_candlestickを複数日扱えるように拡張
            start 09:00 - end 8:55
            '''

            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

            candles = []
            for d in daterange(start_date, end_date + timedelta(1)):
                tmp = pub.get_candlestick(pair, candle_type, d.strftime('%Y%m%d'))['candlestick'][0]['ohlcv']
                if d == start_date:
                    candles = tmp
                else:
                    candles = merge_list(candles, tmp)
            candles = np.array(candles).astype(np.float64)
            return candles

        start_dt = datetime.strptime(start_dt, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_dt, '%Y-%m-%d %H:%M:%S')
        now = datetime.now()
        assert (start_dt <= end_dt)
        assert (start_dt <= now)
        assert (end_dt <= now)

        # _get_candles()では、start_date 09:00 - end_date 08:55 のデータを取得し 09:00 以前のデータは取得しないため
        # start_dt.hour < 9 の場合はデータが不足し、end_dt.hour < 9 の場合は余計にデータを取得することになるため、以下の対策をする
        if start_dt.hour < 9:
            start_date = start_dt.date() - timedelta(1)
        else:
            start_date = start_dt.date()

        if end_dt.hour < 9:
            end_date = end_dt.date() - timedelta(1)
        else:
            end_date = end_dt.date()

        tmp = _get_candles(pair, candle_type, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        candles = []
        for c in tmp:
            dt = datetime.fromtimestamp(c[5] / 1000)
            if (start_dt <= dt) and (dt <= end_dt):
                candles.append(c)
        candles = np.array(candles).astype(np.float64)
        return candles

    def get_active_orders(self, pair, options=None):
        try:
            return prv.get_active_orders(pair, options)
        except Exception as e:
            if parse_error(e) == '20001':
                raise APIAuthenticicationError
            else:
                raise Exception

    def get_trade_history(self, pair, order_count):
        try:
            return prv.get_trade_history(pair, order_count)
        except Exception as e:
            if parse_error(e) == '20001':
                raise APIAuthenticicationError
            else:
                raise Exception

    def get_order(self, pair, order_id):
        try:
            return prv.get_order(pair, order_id)
        except Exception as e:
            if parse_error(e) == '20001':
                raise APIAuthenticicationError
            else:
                raise Exception

    def order(self, pair, price, amount, side, order_type):
        try:
            res = prv.order(pair, price, amount, side, order_type)
        except Exception as e:
            if parse_error(e) == '70009':
                raise MarketOrderLimitError
            elif parse_error(e) == '20001':
                raise APIAuthenticicationError
            else:
                raise Exception

        # 約定が完了するまで待つ
        order_id = res['order_id']
        status = self.get_order(pair, order_id)['status']
        order_time = datetime.fromtimestamp(float(self.get_order(pair, order_id)['ordered_at']) / 1000)
        while (status == 'UNFILLED') or (status == 'PARTIALLY_FILLED'):
            # 5分以上経つとキャンセル
            dt_delta = (datetime.now() - order_time).total_seconds()
            if dt_delta > 300:
                try:
                    self.cancel_order(pair, order_id)
                    logger.debug('cancel order {}, side: {}, dt delta: {}'.format(order_id, side, dt_delta))
                except CancelFailedError:
                    pass
                except Exception:
                    raise Exception
            time.sleep(5)
            status = self.get_order(pair, order_id)['status']
        return self.get_order(pair, order_id)

    def cancel_order(self, pair, order_id):
        try:
            prv.cancel_order(pair, order_id)
        except Exception as e:
            if parse_error(e) == '20001':
                raise APIAuthenticicationError
            elif parse_error(e) == '50010':
                raise CancelFailedError
            else:
                raise Exception

    def cancel_orders(self, pair, order_ids):
        try:
            prv.cancel_orders(pair, order_ids)
        except Exception as e:
            if parse_error(e) == '20001':
                raise APIAuthenticicationError
            elif parse_error(e) == '50010':
                raise CancelFailedError
            else:
                raise Exception

    def get_transactions(self, pair, start_dt=None, end_dt=None):
        transactions = pub.get_transactions(pair)['transactions']

        if (start_dt is None) and (end_dt is None):
            return transactions
        elif (start_dt is not None) and (end_dt is not None):
            transactions_part = []
            for transaction in transactions:
                dt = datetime.fromtimestamp(transaction['executed_at'] / 1000)
                if (start_dt <= dt) and (dt < end_dt):
                    transactions_part.append(transaction)
            return transactions_part

    def get_depth(self, pair):
        return pub.get_depth(pair)