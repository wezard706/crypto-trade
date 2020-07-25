import os
import sys
import logging

root_module = os.path.basename(sys.argv[0])
if root_module == 'tr.py':
    logger = logging.getLogger('crypto')
elif root_module == 'sim.py':
    logger = logging.getLogger('simulate')
else:
    pass

class ModelNaive:
    '''
    現在のローソク足と中期移動平均の位置関係で売買を判定する
    '''

    def __init__(self, reward_upper, loss_lower, update_threshold=0.1, step=0.1):
        self.reward_upper = reward_upper
        self.loss_lower = loss_lower
        self.threshold = update_threshold
        self.step = step
        self.init = {'reward_upper': reward_upper, 'loss_lower': loss_lower, 'update_threshold': update_threshold, 'step': step}

    def reset(self):
        self.reward_upper = self.init['reward_upper']
        self.loss_lower = self.init['loss_lower']
        self.threshold = self.init['update_threshold']
        self.step = self.init['step']

    '''
    def is_exit_long(self, price_now, price_bought):
        reward = price_now - price_bought
        if (reward > self.reward_upper) or (reward < self.loss_lower):
            self.reset()
            return True
        return False

    '''
    def is_exit_short(self, price_now, price_bought):
        reward = price_bought - price_now
        if (reward > self.reward_upper) or (reward < self.loss_lower):
            self.reset()
            return True
        return False

    def is_exit_long(self, price_now, price_bought):
        reward = price_now - price_bought
        if reward > self.reward_upper:
            return [True, 'limit']
        elif reward < self.loss_lower:
            return [True, 'market']
        else:
            return [False, '']

    def update(self, price_now, price_bought, position):
        if position == 'long':
            reward = price_now - price_bought
        else:
            reward = price_bought - price_now
        if reward > self.threshold:
            self.reward_upper = self.reward_upper + self.step
            self.loss_lower = self.loss_lower + self.step
            self.threshold = self.threshold + self.step