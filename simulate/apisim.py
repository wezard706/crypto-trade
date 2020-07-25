from trade.api import APIBase

class APISimulator(APIBase):
    def __init__(self, rm):
        self.rm = rm

    def get_asset(self):
        return self.rm.assets

    def get_ticker(self, pair):
        return self.rm.get_ticker()

    def get_candles(self, pair, candle_type, start_dt, end_dt):
        return self.rm.get_candles(candle_type, start_dt, end_dt)

    def get_active_orders(self, pair, options=None):
        '''
        シミュレータはアクティブな注文なしの前提
        '''
        return {'orders': []}

    def get_trade_history(self, pair, order_count):
        '''
        シミュレータは注文履歴なしの前提
        '''
        return {'trades': []}

    def order(self, pair, order_price, amount, action, order_type):
        # 0.0425は11/19-11/21までのbuyのmean(|注文値-実際値|)
        if action == 'buy':
            price = order_price if order_type == 'limit' else order_price + 0.0425
        elif action == 'sell':
            price = order_price if order_type == 'limit' else order_price - 0.0425
        else:
            import sys
            print('invalid action {}'.format(action))
            sys.exit()
        return price