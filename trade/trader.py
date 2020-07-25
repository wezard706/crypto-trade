import os
import sys
import logging
import pickle
import traceback
import time
from datetime import datetime

from common.utils import dict2str
from trade.exception import InsufficientAssetsError

root_module = os.path.basename(sys.argv[0])
if root_module == 'tr.py':
    logger = logging.getLogger('crypto')
elif root_module == 'sim.py':
    logger = logging.getLogger('simulate')
else:
    pass

class Trader:
    def __init__(self, pair, candle_type, buyer, seller, amount, order_type, asset_lower, api):
        self.api = api
        self.pair = pair
        self.candle_type = candle_type
        self.buyer = buyer
        self.seller = seller
        self.amount = amount
        self.order_type = order_type
        self.asset_lower = asset_lower
        self.long_entries_file = 'trade/tmp/long_entries.pkl'
        self.init_long_entries()

    def init_long_entries(self, wait_time=120):
        if os.path.exists(self.long_entries_file):
            # 注文履歴を読み込む
            with open(self.long_entries_file, 'rb') as f:
                self.long_entries = pickle.load(f)
            logger.debug('load {}'.format(self.long_entries_file))
        else:
            # 注文履歴を空で初期化する
            self.long_entries = {}
            logger.debug('init long_entries by null')

        # activeな注文を確認する
        active_ids_org = [active_order['order_id'] for active_order in self.api.get_active_orders(self.pair)['orders']]

        if len(active_ids_org) == 0:
            return

        time_delta = 0
        start_time = datetime.now()
        # 待ち時間がwait_timeを超える or アクティブな注文が全て約定するまで待つ
        active_ids = active_ids_org
        while (time_delta < wait_time) and (len(active_ids) > 0):
            # activeな注文を更新する
            active_ids = [active_order['order_id'] for active_order in self.api.get_active_orders(self.pair)['orders']]
            time_delta = (datetime.now() - start_time).total_seconds()
            logger.debug('active order is remained: {}'.format(len(active_ids)))
            time.sleep(10)

        # 約定しなかった注文を全てキャンセル
        cancel_ids = [active_order['order_id'] for active_order in self.api.get_active_orders(self.pair)['orders']]
        if len(cancel_ids) > 0:
            self.api.cancel_orders(self.pair, cancel_ids)
            logger.debug('cancel order: {}'.format(cancel_ids))

        # 注文履歴を更新
        for active_id in active_ids_org:
            res_order = self.api.get_order(self.pair, active_id)
            self.update_long_entries(res_order)
        logger.debug('finished update long_entries of active orders')

    def update_long_entries(self, res_order):
        if res_order['side'] == 'buy':
            if (res_order['status'] == 'FULLY_FILLED') or (res_order['status'] == 'CANCELED_PARTIALLY_FILLED'):
                self.long_entries[res_order['order_id']] = res_order
                self.long_entries[res_order['order_id']]['status'] = 'NOT_SOLD'
                with open(self.long_entries_file, 'wb') as f:
                    pickle.dump(self.long_entries, f)
        else:
            '''
            if res_order['status'] == 'FULLY_FILLED':
                # 対象の注文履歴を削除する
                self.long_entries.pop(order_id)
                with open(self.long_entries_file, 'wb') as f:
                    pickle.dump(self.long_entries, f)
            elif res_order['status'] == 'CANCELED_PARTIALLY_FILLED':
                # 約定量を更新する
                self.long_entries[order_id]['executed_amount'] -= float(res_order['executed_amount'])
                self.long_entries[order_id]['status'] = 'NOT_SOLD'
                with open(self.long_entries_file, 'wb') as f:
                    pickle.dump(self.long_entries, f)
            elif res_order['status'] == 'CANCELED_UNFILLED':
                self.long_entries[order_id]['status'] = 'NOT_SOLD'
            '''

            long_entries = self.long_entries.copy()
            if res_order['status'] == 'FULLY_FILLED':
                # 注文履歴を全削除する
                self.long_entries.clear()
                with open(self.long_entries_file, 'wb') as f:
                    pickle.dump(self.long_entries, f)
            elif res_order['status'] == 'CANCELED_PARTIALLY_FILLED':
                # 約定量を更新する
                executed_amount_sell = float(res_order['executed_amount'])
                for order_id, entry in long_entries.items():
                    if executed_amount_sell >= float(entry['executed_amount']):
                        self.long_entries.pop(order_id)
                        executed_amount_sell -= float(entry['executed_amount'])
                    else:
                        self.long_entries[order_id]['executed_amount'] -= executed_amount_sell
                        break
                with open(self.long_entries_file, 'wb') as f:
                    pickle.dump(self.long_entries, f)

    def trade(self, candle):
        # 現在の資産を確認する
        assets = self.api.get_asset()
        if assets < self.asset_lower:
            raise InsufficientAssetsError

        # 板価格を確認する
        ticker_price_buy = self.api.get_ticker(self.pair)['buy']
        ticker_price_sell = self.api.get_ticker(self.pair)['sell']

        # 損切りを確認する
        if len(self.long_entries) > 0:
            price_bought = self.long_entries[sorted(self.long_entries.keys())[-1]]['average_price']
            if self.seller.is_cut_loss(ticker_price_sell, price_bought):
                logger.debug('cut loss, ticker_price_buy: {}, price_bought: {}'.format(ticker_price_buy, price_bought))

                active_sells = [active_order for active_order in self.api.get_active_orders(self.pair)['orders'] if active_order['side'] == 'sell']

                # アクティブな注文はキャンセルする
                if len(active_sells) > 0:
                    cancel_ids = [active_id['order_id'] for active_id in active_sells]
                    self.api.cancel_orders(self.pair, cancel_ids)
                    logger.debug('cancel order: {}'.format(cancel_ids))

                # 買った数を計算する
                total_executed_amount_buy = 0
                for order_id, entry in self.long_entries.items():
                    total_executed_amount_buy += float(entry['executed_amount'])

                # 売りの最安値で注文する
                logger.debug('order sell, pair: {}, price: {}, amount: {}, order type: {}'.format(self.pair, ticker_price_sell, total_executed_amount_buy, 'market'))
                res_order = self.api.order(self.pair, ticker_price_sell, total_executed_amount_buy, 'sell', 'market')
                logger.debug('order finished, ' + dict2str(res_order))

                # 注文履歴を更新
                self.update_long_entries(res_order)

        # 投資指標を計算
        position, score = self.buyer.calc_score(candle)
        logger.debug('trade index, score: {}, position: {}'.format(score, position))

        # 売買する
        if position == 'buy':
            # 買いの最高値で注文する
            logger.debug('order buy, pair: {}, price: {}, amount: {}, order type: {}'.format(self.pair, ticker_price_buy, self.amount, self.order_type))
            res_order = self.api.order(self.pair, ticker_price_buy, self.amount, 'buy', self.order_type)
            logger.debug('order finished, ' + dict2str(res_order))

            # 注文履歴を更新
            self.update_long_entries(res_order)
        elif position == 'sell' and len(self.long_entries) > 0:
            active_sell = [active_order for active_order in self.api.get_active_orders(self.pair)['orders'] if active_order['side'] == 'sell']
            print('{} {}'.format(len(active_sell), self.api.get_active_orders(self.pair)['orders']))
            if len(active_sell) == 0:
                # 買った数を計算する
                total_executed_amount_buy = 0
                for order_id, entry in self.long_entries.items():
                    total_executed_amount_buy += float(entry['executed_amount'])
                # 売りの最安値で注文する
                logger.debug('order sell, pair: {}, price: {}, amount: {}, order type: {}'.format(self.pair, ticker_price_sell, total_executed_amount_buy, self.order_type))
                res_order = self.api.order(self.pair, ticker_price_sell, total_executed_amount_buy, 'sell', self.order_type)
                logger.debug('order finished, ' + dict2str(res_order))

                # 注文履歴を更新
                self.update_long_entries(res_order)

        '''
        long_entries = self.long_entries.copy()
        for order_id, entry in long_entries.items():
            if entry['status'] == 'ACTIVE':
                continue
            price_now = self.api.get_ticker(self.pair)['sell']
            price_bought = float(entry['average_price'])
            is_exit, order_type = self.seller.is_exit_long(float(price_now), price_bought)
            if is_exit:
                logger.debug('order sell, pair: {}, price: {}, amount: {}, order type: {}'.format(self.pair, price_now, self.amount, order_type))
                self.long_entries[order_id]['status'] = 'ACTIVE'
                res_order = self.api.order(self.pair, price_now, float(entry['executed_amount']), 'sell', order_type)
                logger.debug('order finished, ' + dict2str(res_order))

                # 注文履歴を更新
                self.update_long_entries(res_order, order_id)
        '''

        # デバッグ用コメント
        total_executed_amount_buy = 0
        for order_id, entry in self.long_entries.items():
            total_executed_amount_buy += float(entry['executed_amount'])
        logger.debug('current orders, order num: {}, remained amount: {}'.format(len(self.long_entries), total_executed_amount_buy))