import logging
import configparser
from slack_log_handler import SlackLogHandler

inifile = configparser.ConfigParser()
inifile.read('config.ini', 'UTF-8')
alert_url = inifile.get('slack', 'alert_url')

# logger
logger = logging.getLogger('crypto')
logger.setLevel(logging.DEBUG)
format = logging.Formatter('[%(levelname)s] %(asctime)s, %(message)s')
# 標準出力
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(format)
logger.addHandler(stream_handler)
# ファイル出力(debug)
dfile_handler = logging.FileHandler('trade/log/debug.log', 'a')
dfile_handler.setLevel(logging.DEBUG)
dfile_handler.setFormatter(format)
logger.addHandler(dfile_handler)
# ファイル出力(error)
efile_handler = logging.FileHandler('trade/log/error.log', 'a')
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

import os
import sys

import traceback

from trade.params import get_params
from common.utils import dict2str
from trade.tte import TimeTradeEnvironment
from trade.trader import Trader
from trade.model.modellgb_binary import ModelLGBBinary
from trade.model.modelna import ModelNaive
from trade.api import API

def trade():
    params = get_params()
    api = API()

    try:
        logger.info('[params], ' + dict2str(params) + 'initial_assets: ' + str(api.get_asset()))

        # Listenerの追加
        buyer = ModelLGBBinary(params['pair'], api, params['norm_mean'], params['norm_std'], params['lower'], params['upper'], params['loss_lower'])
        seller = ModelLGBBinary(params['pair'], api, params['norm_mean'], params['norm_std'], params['lower'], params['upper'], params['loss_lower'])
        #seller = ModelNaive(params['reward_upper'], params['loss_lower'])
        trader = Trader(params['pair'], params['candle_type'], buyer, seller, params['amount'], params['order_type'], params['asset_lower'], api)
        TimeTradeEnvironment(params['candle_type'], trader, api, params['pair']).run()
    except Exception:
        logger.error(traceback.format_exc())
        logger.debug('process reboot')
        os.execv(sys.executable, [sys.executable] + ['trade/tr.py'])

if __name__=='__main__':
    trade()