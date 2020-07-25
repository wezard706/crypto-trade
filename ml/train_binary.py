import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import gc
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import talib as ta
import itertools
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy.stats import linregress
from multiprocessing import Pool

from trade.api import API
from ml.params import get_params
from common.utils import calc_rmse, plot_confusion_matrix, format_dt, str2dt, dt2str, datetimerange
from ml.create_dataset import create_dataset_api

# logger
import logging
logger = logging.getLogger('ml')
logger.setLevel(logging.DEBUG)
format = logging.Formatter('[%(levelname)s] %(asctime)s, %(message)s')
# 標準出力
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(format)
logger.addHandler(stream_handler)

def train(X_train, y_train, prefix):
    '''
    params_grid = {
        'num_leaves': [8, 32, 128, 256, 1024], # モデルの表現力
        'min_data_in_leaf': [10, 50, 100, 200], # 過学習を調整
        'n_estimators': [100], # 高いと過学習する、learning_rateを下げて調整
        'learning_rate': [0.1] # 低いほどロバスト、n_estimatorsと反比例
    }
    '''

    # start_dt = '2019-01-03 00:00:00'
    # end_dt = '2019-01-26 23:59:59'
    # train_test_split = '2019-01-24 00:00:00'
    # CV: (auc, accuracy, precision, recall) = (0.858233334189654, 0.7877644331524134, 0.7904033436503942, 0.7830772061461774)
    # Test: (auc, accuracy, precision, recall) = (0.8336281416821685, 0.7506944444444444, 0.7648698884758365, 0.7424447451511051)
    # 相関係数: 0.14526641
    params_grid = {
        'num_leaves': [16],  # モデルの表現力
        'min_data_in_leaf': [80],  # 過学習を調整
        'n_estimators': [200],  # 高いと過学習する、learning_rateを下げて調整
        'learning_rate': [0.05]  # 低いほどロバスト、n_estimatorsと反比例
    }

    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    
    best_auc = -1
    best_auc_std = None
    best_accuracy = None
    best_accuracy_std = None
    best_precision = None
    best_precision_std = None
    best_recall = None
    best_recall_std = None
    best_confusion_ravel = None
    best_confusion_ravel_std = None
    best_imp = None
    best_imp_std = None
    best_param = None
    logger.debug('grid search')
    for params in itertools.product(*params_grid.values()):
        param = {k: v for k, v in zip(params_grid.keys(), params)}

        list_auc = []
        list_accuracy = []
        list_precision = []
        list_recall = []
        list_confusion_ravel = []
        list_imp = []
        for i, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
            logger.debug('cv {} / {}'.format(i + 1, n_splits))
            _X_train = X_train.iloc[train_idx]
            _y_train = y_train.iloc[train_idx]
            _X_valid = X_train.iloc[valid_idx]
            _y_valid = y_train.iloc[valid_idx]

            clf = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', is_unbalance=True, nthread=4, **param)
            clf.fit(_X_train.values, _y_train.values, eval_metric='auc')
            pred_valid = clf.predict_proba(_X_valid)[:, 1]
            label_valid = clf.predict(_X_valid)
            list_auc.append(metrics.roc_auc_score(_y_valid, pred_valid))
            tp, fn, fp, tn = confusion_matrix(_y_valid, label_valid).ravel()
            list_accuracy.append((tp + tn) / (tp + fn + fp + tn))
            list_precision.append(tp / (tp + fp))
            list_recall.append(tp / (tp + fn))
            list_confusion_ravel.append((tp, fn, fp, tn))
            list_imp.append(clf.feature_importances_)

            del _X_train, _X_valid, _y_train, _y_valid
            gc.collect()

        mean_auc = np.mean(list_auc)
        std_auc = np.std(list_auc)
        mean_accuracy = np.mean(list_accuracy)
        std_accuracy = np.std(list_accuracy)
        mean_precision = np.mean(list_precision)
        std_precision = np.std(list_precision)
        mean_recall = np.mean(list_recall)
        std_recall = np.std(list_recall)
        mean_confusion_ravel = np.mean(list_confusion_ravel, axis=0)
        std_confusion_ravel = np.std(list_confusion_ravel, axis=0)
        imp_mean = np.mean(list_imp, axis=0)
        imp_std = np.std(list_imp, axis=0)
        logger.debug('{}, auc: {}, accuracy: {}, precision: {}, recall {}'.format(param, mean_auc, mean_accuracy, mean_precision, mean_recall))
        logger.debug('{}'.format(mean_confusion_ravel))
        if best_auc < mean_auc:
            best_auc = mean_auc
            best_auc_std = std_auc
            best_accuracy = mean_accuracy
            best_accuracy_std = std_accuracy
            best_precision = mean_precision
            best_precision_std = std_precision
            best_recall = mean_recall
            best_recall_std = std_recall
            best_confusion_ravel = mean_confusion_ravel
            best_confusion_ravel_std = std_confusion_ravel
            best_imp = imp_mean
            best_imp_std = imp_std
            best_param = param

    logger.debug('CV: (auc, accuracy, precision, recall) = ({}, {}, {}, {})'.format(best_auc, best_accuracy, best_precision, best_recall))
    imp = np.concatenate([best_imp.reshape(len(best_imp), 1), best_imp_std.reshape(len(best_imp_std), 1)], axis=1)
    imp = pd.DataFrame(imp, columns=['mean', 'std'], index=X_train.columns)
    imp = imp.sort_values('mean', ascending=False)
    imp.to_csv('ml/result/{}_importance.csv'.format(prefix))
    #logger.debug('importance: \n {}'.format(imp))

    clf = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', nthread=4, **best_param)
    clf.fit(X_train.values, y_train.values)
    return clf

def calc_change(line1, line2):
    diff = line1 - line2
    change = pd.Series(np.where(diff >= 0, 0.5, -0.5), index=diff.index).diff() # 2階微分（変化点）
    return change

def calc_highests(high, lower=2):
    '''
    天井を求める
    '''
    X = np.arange(len(high))
    # 回帰直線より大きい値（初期値は高値）
    highests = pd.Series(high.reset_index(drop=True), index=X)
    while len(highests) > lower:
        # highestsを元に回帰直線を求める
        reg = linregress(x = highests.index, y = highests)
        reg_line = reg[0] * highests.index + reg[1]
        # highestsを更新する
        tmp = highests.loc[highests > reg_line]
        if len(tmp) == len(highests):
            return None
        highests = tmp
    if len(highests) <= 1:
        return None
    return highests

def calc_lowests(low, lower=2):
    '''
    底値を求める
    '''
    X = np.arange(len(low))
    # 回帰直線より大きい値（初期値は高値）
    lowests = pd.Series(low.reset_index(drop=True), index=X)
    while len(lowests) > lower:
        # highestsを元に回帰直線を求める
        reg = linregress(x=lowests.index, y=lowests)
        reg_line = reg[0] * lowests.index + reg[1]
        # highestsを更新する
        tmp = lowests.loc[lowests < reg_line]
        if len(tmp) == len(lowests):
            return None
        lowests = tmp
    if len(lowests) <= 1:
        return None
    return lowests

def calc_trendlines(candle, w_size=100, stride=30):
    results = []
    starts = range(0, len(candle), stride)
    ends = range(w_size, len(candle), stride)
    X = np.arange(len(candle))
    for w_start, w_end in zip(starts, ends):
        # 天井のトレンドライン
        highests = calc_highests(candle.ix[w_start:w_end, 'high'])
        if highests is not None:
            reg = linregress(x=highests.index + w_start, y=highests)
            #results.append((reg_highest[0], reg_highest[1], trendline))
            results.append({'a': reg[0], 'b': reg[1], 'data': reg[0] * X + reg[1]})
        # 底値のトレンドライン
        lowests = calc_lowests(candle.ix[w_start:w_end, 'low'])
        if lowests is not None:
            reg = linregress(x=lowests.index + w_start, y=lowests)
            results.append({'a': reg[0], 'b': reg[1], 'data': reg[0] * X + reg[1]})
    return results

def calc_trendline_feature(candle, trendlines):
    curr_high = candle.iloc[-1].high
    curr_low = candle.iloc[-1].low

    features = {}
    min_diff_high = 9999
    min_diff_low = 9999
    for trendline in trendlines:
        curr_trend = trendline['data'][-1]
        diff_high = curr_trend - curr_high
        diff_low = curr_trend - curr_low

        if min_diff_high > np.abs(diff_high):
            features['high_a'] = trendline['a']
            features['high_b'] = trendline['b']
            features['high_diff'] = diff_high
            min_diff_high = np.abs(diff_high)
        if min_diff_low > np.abs(diff_low):
            features['low_a'] = trendline['a']
            features['low_b'] = trendline['b']
            features['low_diff'] = diff_low
            min_diff_low = np.abs(diff_low)
    return features

def _extract_feature(candle, params, candle_type, target_dt):
    '''
    前に余分に必要なデータ量: {(stockf_fastk_period_l + stockf_fastk_period_l) * 最大分足 (min)} + window_size
    = (12 + 12) * 5 + 5 = 125 (min)
    '''
    o = candle.open
    h = candle.high
    l = candle.low
    c = candle.close
    v = candle.volume

    # OHLCV
    features = pd.DataFrame()
    features['open'] = o
    features['high'] = h
    features['low'] = l
    features['close'] = c
    features['volume'] = v

    ####################################
    #
    # Momentum Indicator Functions
    #
    ####################################

    # ADX = SUM((+DI - (-DI)) / (+DI + (-DI)), N) / N
    # N — 計算期間
    # SUM (..., N) — N期間の合計
    # +DI — プラスの価格変動の値（positive directional index)
    # -DI — マイナスの価格変動の値（negative directional index）
    # rsi_timeperiod_l=30の場合、30分足で、(30 * 30 / 60(min)) = 15時間必要

    features['adx_s'] = ta.ADX(h, l, c, timeperiod=params['adx_timeperiod_s'])
    features['adx_m'] = ta.ADX(h, l, c, timeperiod=params['adx_timeperiod_m'])
    features['adx_l'] = ta.ADX(h, l, c, timeperiod=params['adx_timeperiod_l'])

    features['adxr_s'] = ta.ADXR(h, l, c, timeperiod=params['adxr_timeperiod_s'])
    features['adxr_m'] = ta.ADXR(h, l, c, timeperiod=params['adxr_timeperiod_m'])
    features['adxr_l'] = ta.ADXR(h, l, c, timeperiod=params['adxr_timeperiod_l'])

    # APO = Shorter Period EMA – Longer Period EMA
    features['apo_s'] = ta.APO(c, fastperiod=params['apo_fastperiod_s'], slowperiod=params['apo_slowperiod_s'], matype=ta.MA_Type.EMA)
    features['apo_m'] = ta.APO(c, fastperiod=params['apo_fastperiod_m'], slowperiod=params['apo_slowperiod_m'], matype=ta.MA_Type.EMA)

    # AroonUp = (N － 過去N日間の最高値からの経過期間) ÷ N × 100
    # AroonDown = (N － 過去N日間の最安値からの経過期間) ÷ N × 100
    # aroon_timeperiod_l=30の場合、30分足で、(30 * 30 / 60(min)) = 15時間必要
    #features['aroondown_s'], features['aroonup_s'] = ta.AROON(h, l, timeperiod=params['aroon_timeperiod_s'])
    #features['aroondown_m'], features['aroonup_m'] = ta.AROON(h, l, timeperiod=params['aroon_timeperiod_m'])
    #features['aroondown_l'], features['aroonup_l'] = ta.AROON(h, l, timeperiod=params['aroon_timeperiod_l'])

    # Aronnオシレーター = AroonUp － AroonDown
    # aroonosc_timeperiod_l=30の場合、30分足で、(30 * 30 / 60(min)) = 15時間必要
    features['aroonosc_s'] = ta.AROONOSC(h, l, timeperiod=params['aroonosc_timeperiod_s'])
    features['aroonosc_m'] = ta.AROONOSC(h, l, timeperiod=params['aroonosc_timeperiod_m'])
    features['aroonosc_l'] = ta.AROONOSC(h, l, timeperiod=params['aroonosc_timeperiod_l'])

    # BOP = (close - open) / (high - low)
    features['bop'] = ta.BOP(o, h, l, c)

    # CCI = (TP - MA) / (0.015 * MD)
    # TP: (高値+安値+終値) / 3
    # MA: TPの移動平均
    # MD: 平均偏差 = ((MA - TP1) + (MA - TP2) + ...) / N
    features['cci_s'] = ta.CCI(h, l, c, timeperiod=params['cci_timeperiod_s'])
    features['cci_m'] = ta.CCI(h, l, c, timeperiod=params['cci_timeperiod_m'])
    features['cci_l'] = ta.CCI(h, l, c, timeperiod=params['cci_timeperiod_l'])

    # CMO - Chande Momentum Oscillator
    #features['cmo_s'] = ta.CMO(c, timeperiod=params['cmo_timeperiod_s'])
    #features['cmo_m'] = ta.CMO(c, timeperiod=params['cmo_timeperiod_m'])
    #features['cmo_l'] = ta.CMO(c, timeperiod=params['cmo_timeperiod_l'])

    # DX - Directional Movement Index
    features['dx_s'] = ta.DX(h, l, c, timeperiod=params['dx_timeperiod_s'])
    features['dx_m'] = ta.DX(h, l, c, timeperiod=params['dx_timeperiod_m'])
    features['dx_l'] = ta.DX(h, l, c, timeperiod=params['dx_timeperiod_l'])

    # MACD＝基準線－相対線
    # 基準線（EMA）：過去12日（週・月）間の終値指数平滑平均
    # 相対線（EMA）：過去26日（週・月）間の終値指数平滑平均
    # https://www.sevendata.co.jp/shihyou/technical/macd.html
    # macd_slowperiod_m = 30 の場合30分足で（(30 + macd_signalperiod_m) * 30）/ 60 = 16.5時間必要（macd_signalperiod_m=3の時）
    macd, macdsignal, macdhist = ta.MACDEXT(c, fastperiod=params['macd_fastperiod_s'],
                                            slowperiod=params['macd_slowperiod_s'],
                                            signalperiod=params['macd_signalperiod_s'],
                                            fastmatype=ta.MA_Type.EMA, slowmatype=ta.MA_Type.EMA,
                                            signalmatype=ta.MA_Type.EMA)
    change_macd = calc_change(macd, macdsignal)
    change_macd.index = macd.index
    features['macd_s'] = macd
    features['macdsignal_s'] = macdsignal
    features['macdhist_s'] = macdhist
    features['change_macd_s'] = change_macd
    macd, macdsignal, macdhist = ta.MACDEXT(c, fastperiod=params['macd_fastperiod_m'],
                                            slowperiod=params['macd_slowperiod_m'],
                                            signalperiod=params['macd_signalperiod_m'],
                                            fastmatype=ta.MA_Type.EMA, slowmatype=ta.MA_Type.EMA,
                                            signalmatype=ta.MA_Type.EMA)
    change_macd = calc_change(macd, macdsignal)
    change_macd.index = macd.index
    features['macd_m'] = macd
    features['macdsignal_m'] = macdsignal
    features['macdhist_m'] = macdhist
    features['change_macd_m'] = change_macd

    # MFI - Money Flow Index
    features['mfi_s'] = ta.MFI(h, l, c, v, timeperiod=params['mfi_timeperiod_s'])
    features['mfi_m'] = ta.MFI(h, l, c, v, timeperiod=params['mfi_timeperiod_m'])
    features['mfi_l'] = ta.MFI(h, l, c, v, timeperiod=params['mfi_timeperiod_l'])

    # MINUS_DI - Minus Directional Indicator
    features['minus_di_s'] = ta.MINUS_DI(h, l, c, timeperiod=params['minus_di_timeperiod_s'])
    features['minus_di_m'] = ta.MINUS_DI(h, l, c, timeperiod=params['minus_di_timeperiod_m'])
    features['minus_di_l'] = ta.MINUS_DI(h, l, c, timeperiod=params['minus_di_timeperiod_l'])

    # MINUS_DM - Minus Directional Movement
    features['minus_dm_s'] = ta.MINUS_DM(h, l, timeperiod=params['minus_dm_timeperiod_s'])
    features['minus_dm_m'] = ta.MINUS_DM(h, l, timeperiod=params['minus_dm_timeperiod_m'])
    features['minus_dm_l'] = ta.MINUS_DM(h, l, timeperiod=params['minus_dm_timeperiod_l'])

    # MOM - Momentum
    features['mom_s'] = ta.MOM(c, timeperiod=params['mom_timeperiod_s'])
    features['mom_m'] = ta.MOM(c, timeperiod=params['mom_timeperiod_m'])
    features['mom_l'] = ta.MOM(c, timeperiod=params['mom_timeperiod_l'])

    # PLUS_DI - Minus Directional Indicator
    features['plus_di_s'] = ta.PLUS_DI(h, l, c, timeperiod=params['plus_di_timeperiod_s'])
    features['plus_di_m'] = ta.PLUS_DI(h, l, c, timeperiod=params['plus_di_timeperiod_m'])
    features['plus_di_l'] = ta.PLUS_DI(h, l, c, timeperiod=params['plus_di_timeperiod_l'])

    # PLUS_DM - Minus Directional Movement
    features['plus_dm_s'] = ta.PLUS_DM(h, l, timeperiod=params['plus_dm_timeperiod_s'])
    features['plus_dm_m'] = ta.PLUS_DM(h, l, timeperiod=params['plus_dm_timeperiod_m'])
    features['plus_dm_l'] = ta.PLUS_DM(h, l, timeperiod=params['plus_dm_timeperiod_l'])

    # PPO - Percentage Price Oscillator
    #features['ppo_s'] = ta.PPO(c, fastperiod=params['ppo_fastperiod_s'], slowperiod=params['ppo_slowperiod_s'], matype=ta.MA_Type.EMA)
    #features['ppo_m'] = ta.PPO(c, fastperiod=params['ppo_fastperiod_m'], slowperiod=params['ppo_slowperiod_m'], matype=ta.MA_Type.EMA)

    # ROC - Rate of change : ((price/prevPrice)-1)*100
    features['roc_s'] = ta.ROC(c, timeperiod=params['roc_timeperiod_s'])
    features['roc_m'] = ta.ROC(c, timeperiod=params['roc_timeperiod_m'])
    features['roc_l'] = ta.ROC(c, timeperiod=params['roc_timeperiod_l'])

    # ROCP = (price-prevPrice) / prevPrice
    # http://www.tadoc.org/indicator/ROCP.htm
    # rocp_timeperiod_l = 30 の場合、30分足で(30 * 30) / 60 = 15時間必要
    rocp = ta.ROCP(c, timeperiod=params['rocp_timeperiod_s'])
    change_rocp = calc_change(rocp, pd.Series(np.zeros(len(candle)), index=candle.index))
    change_rocp.index = rocp.index
    features['rocp_s'] = rocp
    features['change_rocp_s'] = change_rocp
    rocp = ta.ROCP(c, timeperiod=params['rocp_timeperiod_m'])
    change_rocp = calc_change(rocp, pd.Series(np.zeros(len(candle)), index=candle.index))
    change_rocp.index = rocp.index
    features['rocp_m'] = rocp
    features['change_rocp_m'] = change_rocp
    rocp = ta.ROCP(c, timeperiod=params['rocp_timeperiod_l'])
    change_rocp = calc_change(rocp, pd.Series(np.zeros(len(candle)), index=candle.index))
    change_rocp.index = rocp.index
    features['rocp_l'] = rocp
    features['change_rocp_l'] = change_rocp

    # ROCR - Rate of change ratio: (price/prevPrice)
    features['rocr_s'] = ta.ROCR(c, timeperiod=params['rocr_timeperiod_s'])
    features['rocr_m'] = ta.ROCR(c, timeperiod=params['rocr_timeperiod_m'])
    features['rocr_l'] = ta.ROCR(c, timeperiod=params['rocr_timeperiod_l'])

    # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
    features['rocr100_s'] = ta.ROCR100(c, timeperiod=params['rocr100_timeperiod_s'])
    features['rocr100_m'] = ta.ROCR100(c, timeperiod=params['rocr100_timeperiod_m'])
    features['rocr100_l'] = ta.ROCR100(c, timeperiod=params['rocr100_timeperiod_l'])

    # RSI = (100 * a) / (a + b) (a: x日間の値上がり幅の合計, b: x日間の値下がり幅の合計)
    # https://www.sevendata.co.jp/shihyou/technical/rsi.html
    # rsi_timeperiod_l=30の場合、30分足で、(30 * 30 / 60(min)) = 15時間必要
    #features['rsi_s'] = ta.RSI(c, timeperiod=params['rsi_timeperiod_s'])
    #features['rsi_m'] = ta.RSI(c, timeperiod=params['rsi_timeperiod_m'])
    #features['rsi_l'] = ta.RSI(c, timeperiod=params['rsi_timeperiod_l'])


    # FASTK(KPeriod) = 100 * (Today's Close - LowestLow) / (HighestHigh - LowestLow)
    # FASTD(FastDperiod) = MA Smoothed FASTK over FastDperiod
    # http://www.tadoc.org/indicator/STOCHF.htm
    # stockf_fastk_period_l=30の場合30分足で、(((30 + 30) * 30) / 60(min)) = 30時間必要 (LowestLowが移動平均の30分余分に必要なので60period余分に計算する)
    fastk, fastd = ta.STOCHF(h, l, c, fastk_period=params['stockf_fastk_period_s'], fastd_period=params['stockf_fastd_period_s'], fastd_matype=ta.MA_Type.EMA)
    change_stockf = calc_change(fastk, fastd)
    change_stockf.index = fastk.index
    features['fastk_s'] = fastk
    features['fastd_s'] = fastd
    features['fast_change_s'] = change_stockf
    fastk, fastd = ta.STOCHF(h, l, c, fastk_period=params['stockf_fastk_period_m'], fastd_period=params['stockf_fastd_period_m'], fastd_matype=ta.MA_Type.EMA)
    change_stockf = calc_change(fastk, fastd)
    change_stockf.index = fastk.index
    features['fastk_m'] = fastk
    features['fastd_m'] = fastd
    features['fast_change_m'] = change_stockf
    fastk, fastd = ta.STOCHF(h, l, c, fastk_period=params['stockf_fastk_period_l'], fastd_period=params['stockf_fastk_period_l'], fastd_matype=ta.MA_Type.EMA)
    change_stockf = calc_change(fastk, fastd)
    change_stockf.index = fastk.index
    features['fastk_l'] = fastk
    features['fastd_l'] = fastd
    features['fast_change_l'] = change_stockf

    # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    features['trix_s'] = ta.TRIX(c, timeperiod=params['trix_timeperiod_s'])
    features['trix_m'] = ta.TRIX(c, timeperiod=params['trix_timeperiod_m'])
    features['trix_l'] = ta.TRIX(c, timeperiod=params['trix_timeperiod_l'])

    # ULTOSC - Ultimate Oscillator
    features['ultosc_s'] = ta.ULTOSC(h, l, c, timeperiod1=params['ultosc_timeperiod_s1'], timeperiod2=params['ultosc_timeperiod_s2'], timeperiod3=params['ultosc_timeperiod_s3'])

    # WILLR = (当日終値 - N日間の最高値) / (N日間の最高値 - N日間の最安値)× 100
    # https://inet-sec.co.jp/study/technical-manual/williamsr/
    # willr_timeperiod_l=30の場合30分足で、(30 * 30 / 60) = 15時間必要
    features['willr_s'] = ta.WILLR(h, l, c, timeperiod=params['willr_timeperiod_s'])
    features['willr_m'] = ta.WILLR(h, l, c, timeperiod=params['willr_timeperiod_m'])
    features['willr_l'] = ta.WILLR(h, l, c, timeperiod=params['willr_timeperiod_l'])

    ####################################
    #
    # Volume Indicator Functions
    #
    ####################################

    # Volume Indicator Functions
    # slowperiod_adosc_s = 10の場合、30分足で(10 * 30) / 60 = 5時間必要
    features['ad'] = ta.AD(h, l, c, v)
    features['adosc_s'] = ta.ADOSC(h, l, c, v, fastperiod=params['fastperiod_adosc_s'], slowperiod=params['slowperiod_adosc_s'])
    features['obv'] = ta.OBV(c, v)

    ####################################
    #
    # Volatility Indicator Functions
    #
    ####################################

    # ATR - Average True Range
    features['atr_s'] = ta.ATR(h, l, c, timeperiod=params['atr_timeperiod_s'])
    features['atr_m'] = ta.ATR(h, l, c, timeperiod=params['atr_timeperiod_m'])
    features['atr_l'] = ta.ATR(h, l, c, timeperiod=params['atr_timeperiod_l'])

    # NATR - Normalized Average True Range
    #features['natr_s'] = ta.NATR(h, l, c, timeperiod=params['natr_timeperiod_s'])
    #features['natr_m'] = ta.NATR(h, l, c, timeperiod=params['natr_timeperiod_m'])
    #features['natr_l'] = ta.NATR(h, l, c, timeperiod=params['natr_timeperiod_l'])

    # TRANGE - True Range
    features['trange'] = ta.TRANGE(h, l, c)

    ####################################
    #
    # Price Transform Functions
    #
    ####################################

    features['avgprice'] = ta.AVGPRICE(o, h, l, c)
    features['medprice'] = ta.MEDPRICE(h, l)
    #features['typprice'] = ta.TYPPRICE(h, l, c)
    #features['wclprice'] = ta.WCLPRICE(h, l, c)

    ####################################
    #
    # Cycle Indicator Functions
    #
    ####################################

    #features['ht_dcperiod'] = ta.HT_DCPERIOD(c)
    #features['ht_dcphase'] = ta.HT_DCPHASE(c)
    #features['inphase'], features['quadrature'] = ta.HT_PHASOR(c)
    #features['sine'], features['leadsine'] = ta.HT_SINE(c)
    features['integer'] = ta.HT_TRENDMODE(c)

    ####################################
    #
    # Statistic Functions
    #
    ####################################

    # BETA - Beta

    features['beta_s'] = ta.BETA(h, l, timeperiod=params['beta_timeperiod_s'])
    features['beta_m'] = ta.BETA(h, l, timeperiod=params['beta_timeperiod_m'])
    features['beta_l'] = ta.BETA(h, l, timeperiod=params['beta_timeperiod_l'])

    # CORREL - Pearson's Correlation Coefficient (r)
    #features['correl_s'] = ta.CORREL(h, l, timeperiod=params['correl_timeperiod_s'])
    #features['correl_m'] = ta.CORREL(h, l, timeperiod=params['correl_timeperiod_m'])
    #features['correl_l'] = ta.CORREL(h, l, timeperiod=params['correl_timeperiod_l'])

    # LINEARREG - Linear Regression
    #features['linearreg_s'] = ta.LINEARREG(c, timeperiod=params['linearreg_timeperiod_s'])
    #features['linearreg_m'] = ta.LINEARREG(c, timeperiod=params['linearreg_timeperiod_m'])
    #features['linearreg_l'] = ta.LINEARREG(c, timeperiod=params['linearreg_timeperiod_l'])

    # LINEARREG_ANGLE - Linear Regression Angle
    features['linearreg_angle_s'] = ta.LINEARREG_ANGLE(c, timeperiod=params['linearreg_angle_timeperiod_s'])
    features['linearreg_angle_m'] = ta.LINEARREG_ANGLE(c, timeperiod=params['linearreg_angle_timeperiod_m'])
    features['linearreg_angle_l'] = ta.LINEARREG_ANGLE(c, timeperiod=params['linearreg_angle_timeperiod_l'])

    # LINEARREG_INTERCEPT - Linear Regression Intercept
    features['linearreg_intercept_s'] = ta.LINEARREG_INTERCEPT(c, timeperiod=params['linearreg_intercept_timeperiod_s'])
    features['linearreg_intercept_m'] = ta.LINEARREG_INTERCEPT(c, timeperiod=params['linearreg_intercept_timeperiod_m'])
    features['linearreg_intercept_l'] = ta.LINEARREG_INTERCEPT(c, timeperiod=params['linearreg_intercept_timeperiod_l'])

    # LINEARREG_SLOPE - Linear Regression Slope
    features['linearreg_slope_s'] = ta.LINEARREG_SLOPE(c, timeperiod=params['linearreg_slope_timeperiod_s'])
    features['linearreg_slope_m'] = ta.LINEARREG_SLOPE(c, timeperiod=params['linearreg_slope_timeperiod_m'])
    features['linearreg_slope_l'] = ta.LINEARREG_SLOPE(c, timeperiod=params['linearreg_slope_timeperiod_l'])

    # STDDEV - Standard Deviation
    features['stddev_s'] = ta.STDDEV(c, timeperiod=params['stddev_timeperiod_s'], nbdev=1)
    features['stddev_m'] = ta.STDDEV(c, timeperiod=params['stddev_timeperiod_m'], nbdev=1)
    features['stddev_l'] = ta.STDDEV(c, timeperiod=params['stddev_timeperiod_l'], nbdev=1)

    # TSF - Time Series Forecast
    features['tsf_s'] = ta.TSF(c, timeperiod=params['tsf_timeperiod_s'])
    features['tsf_m'] = ta.TSF(c, timeperiod=params['tsf_timeperiod_m'])
    features['tsf_l'] = ta.TSF(c, timeperiod=params['tsf_timeperiod_l'])

    # VAR - Variance
    #features['var_s'] = ta.VAR(c, timeperiod=params['var_timeperiod_s'], nbdev=1)
    #features['var_m'] = ta.VAR(c, timeperiod=params['var_timeperiod_m'], nbdev=1)
    #features['var_l'] = ta.VAR(c, timeperiod=params['var_timeperiod_l'], nbdev=1)

    # ボリンジャーバンド
    # bbands_timeperiod_l = 30の場合、30分足で(30 * 30) / 60 = 15時間必要
    bb_upper, bb_middle, bb_lower = ta.BBANDS(c, timeperiod=params['bbands_timeperiod_s'],
                                              nbdevup=params['bbands_nbdevup_s'], nbdevdn=params['bbands_nbdevdn_s'],
                                              matype=ta.MA_Type.EMA)
    bb_trend1 = pd.Series(np.zeros(len(candle)), index=candle.index)
    bb_trend1[c > bb_upper] = 1
    bb_trend1[c < bb_lower] = -1
    bb_trend2 = pd.Series(np.zeros(len(candle)), index=candle.index)
    bb_trend2[c > bb_middle] = 1
    bb_trend2[c < bb_middle] = -1
    features['bb_upper_s'] = bb_upper
    features['bb_middle_s'] = bb_middle
    features['bb_lower_s'] = bb_lower
    features['bb_trend1_s'] = bb_trend1
    features['bb_trend2_s'] = bb_trend2
    bb_upper, bb_middle, bb_lower = ta.BBANDS(c, timeperiod=params['bbands_timeperiod_m'],
                                              nbdevup=params['bbands_nbdevup_m'], nbdevdn=params['bbands_nbdevdn_m'],
                                              matype=ta.MA_Type.EMA)
    bb_trend1 = pd.Series(np.zeros(len(candle)), index=candle.index)
    bb_trend1[c > bb_upper] = 1
    bb_trend1[c < bb_lower] = -1
    bb_trend2 = pd.Series(np.zeros(len(candle)), index=candle.index)
    bb_trend2[c > bb_middle] = 1
    bb_trend2[c < bb_middle] = -1
    features['bb_upper_m'] = bb_upper
    features['bb_middle_m'] = bb_middle
    features['bb_lower_m'] = bb_lower
    features['bb_trend1_m'] = bb_trend1
    features['bb_trend2_m'] = bb_trend2
    bb_upper, bb_middle, bb_lower = ta.BBANDS(c, timeperiod=params['bbands_timeperiod_l'],
                                              nbdevup=params['bbands_nbdevup_l'], nbdevdn=params['bbands_nbdevdn_l'],
                                              matype=ta.MA_Type.EMA)
    bb_trend1 = pd.Series(np.zeros(len(candle)), index=candle.index)
    bb_trend1[c > bb_upper] = 1
    bb_trend1[c < bb_lower] = -1
    bb_trend2 = pd.Series(np.zeros(len(candle)), index=candle.index)
    bb_trend2[c > bb_middle] = 1
    bb_trend2[c < bb_middle] = -1
    features['bb_upper_l'] = bb_upper
    features['bb_middle_l'] = bb_middle
    features['bb_lower_l'] = bb_lower
    features['bb_trend1_l'] = bb_trend1
    features['bb_trend2_l'] = bb_trend2

    # ローソク足
    features['CDL2CROWS'] = ta.CDL2CROWS(o, h, l, c)
    features['CDL3BLACKCROWS'] = ta.CDL3BLACKCROWS(o, h, l, c)
    features['CDL3INSIDE'] = ta.CDL3INSIDE(o, h, l, c)
    features['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(o, h, l, c)
    features['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(o, h, l, c)
    features['CDL3STARSINSOUTH'] = ta.CDL3STARSINSOUTH(o, h, l, c)
    features['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(o, h, l, c)
    features['CDLABANDONEDBABY'] = ta.CDLABANDONEDBABY(o, h, l, c, penetration=0)
    features['CDLADVANCEBLOCK'] = ta.CDLADVANCEBLOCK(o, h, l, c)
    features['CDLBELTHOLD'] = ta.CDLBELTHOLD(o, h, l, c)
    features['CDLBREAKAWAY'] = ta.CDLBREAKAWAY(o, h, l, c)
    features['CDLCLOSINGMARUBOZU'] = ta.CDLCLOSINGMARUBOZU(o, h, l, c)
    features['CDLCONCEALBABYSWALL'] = ta.CDLCONCEALBABYSWALL(o, h, l, c)
    features['CDLCOUNTERATTACK'] = ta.CDLCOUNTERATTACK(o, h, l, c)
    features['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(o, h, l, c, penetration=0)
    #features['CDLDOJI'] = ta.CDLDOJI(o, h, l, c)
    features['CDLDOJISTAR'] = ta.CDLDOJISTAR(o, h, l, c)
    features['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(o, h, l, c)
    features['CDLENGULFING'] = ta.CDLENGULFING(o, h, l, c)
    features['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(o, h, l, c, penetration=0)
    features['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(o, h, l, c, penetration=0)
    #features['CDLGAPSIDESIDEWHITE'] = ta.CDLGAPSIDESIDEWHITE(o, h, l, c)
    features['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(o, h, l, c)
    features['CDLHAMMER'] = ta.CDLHAMMER(o, h, l, c)
    #features['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(o, h, l, c)
    features['CDLHARAMI'] = ta.CDLHARAMI(o, h, l, c)
    features['CDLHARAMICROSS'] = ta.CDLHARAMICROSS(o, h, l, c)
    features['CDLHIGHWAVE'] = ta.CDLHIGHWAVE(o, h, l, c)
    #features['CDLHIKKAKE'] = ta.CDLHIKKAKE(o, h, l, c)
    features['CDLHIKKAKEMOD'] = ta.CDLHIKKAKEMOD(o, h, l, c)
    features['CDLHOMINGPIGEON'] = ta.CDLHOMINGPIGEON(o, h, l, c)
    #features['CDLIDENTICAL3CROWS'] = ta.CDLIDENTICAL3CROWS(o, h, l, c)
    features['CDLINNECK'] = ta.CDLINNECK(o, h, l, c)
    #features['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(o, h, l, c)
    features['CDLKICKING'] = ta.CDLKICKING(o, h, l, c)
    features['CDLKICKINGBYLENGTH'] = ta.CDLKICKINGBYLENGTH(o, h, l, c)
    features['CDLLADDERBOTTOM'] = ta.CDLLADDERBOTTOM(o, h, l, c)
    #features['CDLLONGLEGGEDDOJI'] = ta.CDLLONGLEGGEDDOJI(o, h, l, c)
    features['CDLMARUBOZU'] = ta.CDLMARUBOZU(o, h, l, c)
    #features['CDLMATCHINGLOW'] = ta.CDLMATCHINGLOW(o, h, l, c)
    features['CDLMATHOLD'] = ta.CDLMATHOLD(o, h, l, c, penetration=0)
    features['CDLMORNINGDOJISTAR'] = ta.CDLMORNINGDOJISTAR(o, h, l, c, penetration=0)
    features['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(o, h, l, c, penetration=0)
    features['CDLONNECK'] = ta.CDLONNECK(o, h, l, c)
    features['CDLPIERCING'] = ta.CDLPIERCING(o, h, l, c)
    features['CDLRICKSHAWMAN'] = ta.CDLRICKSHAWMAN(o, h, l, c)
    features['CDLRISEFALL3METHODS'] = ta.CDLRISEFALL3METHODS(o, h, l, c)
    features['CDLSEPARATINGLINES'] = ta.CDLSEPARATINGLINES(o, h, l, c)
    #features['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(o, h, l, c)
    features['CDLSHORTLINE'] = ta.CDLSHORTLINE(o, h, l, c)
    #features['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(o, h, l, c)
    features['CDLSTALLEDPATTERN'] = ta.CDLSTALLEDPATTERN(o, h, l, c)
    features['CDLSTICKSANDWICH'] = ta.CDLSTICKSANDWICH(o, h, l, c)
    features['CDLTAKURI'] = ta.CDLTAKURI(o, h, l, c)
    features['CDLTASUKIGAP'] = ta.CDLTASUKIGAP(o, h, l, c)
    features['CDLTHRUSTING'] = ta.CDLTHRUSTING(o, h, l, c)
    features['CDLTRISTAR'] = ta.CDLTRISTAR(o, h, l, c)
    features['CDLUNIQUE3RIVER'] = ta.CDLUNIQUE3RIVER(o, h, l, c)
    features['CDLUPSIDEGAP2CROWS'] = ta.CDLUPSIDEGAP2CROWS(o, h, l, c)
    features['CDLXSIDEGAP3METHODS'] = ta.CDLXSIDEGAP3METHODS(o, h, l, c)

    '''
    # トレンドライン
    for dt in datetimerange(candle.index[0], candle.index[-1] + timedelta(minutes=1)):
        start_dt = (dt - timedelta(minutes=130)).strftime('%Y-%m-%d %H:%M:%S')
        end_dt = dt.strftime('%Y-%m-%d %H:%M:%S')
        tmp = candle.loc[(start_dt <= candle.index) & (candle.index <= end_dt)]
        for w_size, stride in [(15, 5), (30, 10), (60, 10), (120, 10)]:
        # for w_size, stride in [(120, 10)]:
            trendlines = calc_trendlines(tmp, w_size, stride)
            if len(trendlines) == 0:
                continue
            trendline_feature = calc_trendline_feature(tmp, trendlines)

            print('{}-{} {} {} {}'.format(dt - timedelta(minutes=130), dt, trendline_feature['high_a'], trendline_feature['high_b'], trendline_feature['high_diff']))

            features.loc[features.index == end_dt, 'trendline_high_a_{}'.format(w_size)] = trendline_feature['high_a']
            features.loc[features.index == end_dt, 'trendline_high_b_{}'.format(w_size)] = trendline_feature['high_b']
            features.loc[features.index == end_dt, 'trendline_high_diff_{}'.format(w_size)] = trendline_feature['high_diff']
            features.loc[features.index == end_dt, 'trendline_low_a_{}'.format(w_size)] = trendline_feature['low_a']
            features.loc[features.index == end_dt, 'trendline_low_b_{}'.format(w_size)] = trendline_feature['low_b']
            features.loc[features.index == end_dt, 'trendline_low_diff_{}'.format(w_size)] = trendline_feature['low_diff']
    '''

    window = 5
    features_ext = features
    for w in range(window):
        tmp = features.shift(periods=60 * (w + 1), freq='S')
        tmp.columns = [c + '_' + str(w + 1) + 'w' for c in features.columns]
        features_ext = pd.concat([features_ext, tmp], axis=1)
    
    if candle_type == '5min':
        features_ext = features_ext.shift(periods=300, freq='S')
        features_ext = features_ext.fillna(method='ffill')
    features_ext = features_ext[features_ext.index == target_dt]
    return features_ext

def wrapper(args):
    return _extract_feature(*args)

def multi_process(sampleList):
    p = Pool(8)
    output = p.map(wrapper, sampleList)
    p.close()
    return output

def extract_feature(start_dt, end_dt, pair, candle_types, df_all=None):
    features_all = None
    for candle_type in candle_types:
        params = get_params(candle_type)
        if df_all is not None:
            logger.debug('candle type: {}'.format(candle_type))
            candles = df_all[candle_type]
            candles.index = candles.timestamp.map(lambda x: datetime.fromtimestamp(x / 1000))
            # 時刻ごとに特徴量を算出（並列処理）
            args = [(candles[(d - timedelta(minutes=130) <= candles.index) & (candles.index <= d)], params, candle_type, d) for d in datetimerange(str2dt(start_dt), str2dt(end_dt) + timedelta(minutes=1))]
            tmp_features = multi_process(args)

            # 必要な時間のみ抽出
            features = None
            dts = [d for d in datetimerange(str2dt(start_dt), str2dt(end_dt) + timedelta(minutes=1))]
            for dt, tmp_feature in zip(dts, tmp_features):
                feature = tmp_feature[tmp_feature.index == dt]
                if features is None:
                    features = feature
                else:
                    features = pd.concat([features, feature])

            del tmp_features
            gc.collect()
        else:
            start_dt_ext = (datetime.strptime(start_dt, '%Y-%m-%d %H:%M:%S') - timedelta(minutes=130)).strftime('%Y-%m-%d %H:%M:%S')
            candles = API().get_candles(pair, candle_type=candle_type, start_dt=start_dt_ext, end_dt=end_dt)
            candles = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
            candles.index = candles.timestamp.map(lambda x: datetime.fromtimestamp(x / 1000))
            candles.to_csv('candles_{}_{}.csv'.format(end_dt, candle_type))
            features = _extract_feature(candles, params, candle_type, end_dt)
            features.to_csv('features_{}_{}.csv'.format(end_dt, candle_type))

        '''
        features = features.loc[(start_dt <= features.index) & (features.index <= end_dt)]
        '''
        features.columns = [c + '_' + candle_type for c in features.columns]

        if features_all is None:
            features_all = features
        else:
            features_all = pd.concat([features_all, features], axis=1)
            features_all = features_all.fillna(method='ffill')

        del features
        gc.collect()
    return features_all

def create_target(df, start_dt, end_dt):
    candles = pd.DataFrame(df, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
    candles.index = candles.timestamp.map(lambda x: datetime.fromtimestamp(x / 1000))
    target = pd.Series(np.where(-ta.TEMA(candles.close, timeperiod=5).diff(-1) > 0, 1, 0), index=candles.index)
    target = target.loc[(target.index >= start_dt) & (target.index <= end_dt)]
    return target

'''
def get_assets_change(z_scores, prices, dts, threshold):
    # 資産推移
    trade_num = 0
    queue = []
    assets = []
    for dt, price_buy, price_sell, z_score in zip(dts, prices['buy'], prices['sell'], z_scores):
        profit = 0
        if z_score > threshold:
            trade_num += 1
            queue.append(price_sell)
            print('{} buy {} {} {}'.format(dt, price_sell, profit, z_score))
        #elif z_score < -threshold:
        if len(queue) > 0:
            if queue[-1] - price_sell < -0.01:
            #if len(queue) > 0:
                for price in queue:
                    profit += price_buy - price
                    print('{} sell {} {} {}'.format(dt, price_buy, profit, z_score))
                queue = []

        if len(assets) > 0:
            assets.append(assets[-1] + profit)
        else:
            assets.append(profit)
    return assets
'''

def get_assets_change(z_scores, prices, dts, threshold_upper, threshold_lower):
    # 資産推移
    trade_num = 0
    queue = []
    assets = []
    for dt, ticker_price_buy, ticker_price_sell, z_score in zip(dts, prices['buy'], prices['sell'], z_scores):
        profit = 0
        if z_score > threshold_upper:
            trade_num += 1
            threshold_exit = ticker_price_buy - ticker_price_buy
            queue.append((ticker_price_buy, threshold_exit))
            #print('{} buy {} {} {}'.format(dt, ticker_price_sell, profit, z_score))
        elif z_score < threshold_lower:
            if len(queue) > 0:
                for purchase_price, threshold_exit in queue:
                    profit += ticker_price_sell - purchase_price
                    #print('{} sell {} = {} - {} {}'.format(dt, profit, ticker_price_buy, purchase_price, z_score, len(queue)))
                queue = []
        '''
        if len(queue) > 0:
            for price, exit_lower in queue:
                if price - price_buy < 0.01:
                    profit += price_buy - price
        '''

        if len(assets) > 0:
            assets.append(assets[-1] + profit)
        else:
            assets.append(profit)
    return assets

def main():
    is_create_data = False
    is_feature_extraction = False
    is_train = False
    pair = 'xrp_jpy'
    start_dt = '2019-01-01 00:00:00'
    end_dt = '2019-02-02 17:59:59'
    train_test_split = '2019-02-01 00:00:00'
    threshold = 0.1

    candle_types = ['1min', '5min']
    #candle_types = ['1min']

    # データを取得
    candles_all = {}
    prefix = '{}_{}'.format(format_dt(start_dt, '%Y-%m-%d %H:%M:%S', '%Y%m%d%H%M'), format_dt(end_dt, '%Y-%m-%d %H:%M:%S', '%Y%m%d%H%M'))
    if is_create_data:
        start_dt_create_data = dt2str(str2dt(start_dt) - timedelta(minutes=130))
        for candle_type in candle_types:
            candles_all[candle_type] = create_dataset_api(start_dt_create_data, end_dt, pair, candle_type)
            with open('ml/input/{}_candles_api_{}.pkl'.format(prefix, candle_type), 'wb') as f:
                pickle.dump(candles_all[candle_type], f)
    else:
        for candle_type in candle_types:
            with open('ml/input/{}_candles_api_{}.pkl'.format(prefix, candle_type), 'rb') as f:
                candles_all[candle_type] = pickle.load(f)

    if is_feature_extraction:
        features = extract_feature(start_dt, end_dt, 'xrp_jpy', candle_types, df_all=candles_all)
        features.to_pickle('ml/input/{}_features.pkl'.format(prefix))
    else:
        with open('ml/input/{}_features.pkl'.format(prefix), 'rb') as f:
            features = pickle.load(f)

    # 正解データを作成
    target = create_target(candles_all['1min'], start_dt, end_dt)
    data = pd.concat([features, target], axis=1)
    data.columns = features.columns.tolist() + ['target']
    data = data.loc[~data.target.isnull()]
    #data_pos = data[data['target'] == 0]
    #data_neg = data[data['target'] == 1]
    #data = pd.concat([data_pos.sample(n=data_neg.shape[0], random_state=0), data_neg])

    train_mask = train_test_split > data.index
    test_mask = train_test_split <= data.index
    X_train = data.loc[train_mask, [c for c in data.columns if c != 'target']]
    y_train = data.loc[train_mask, ['target']]
    X_test = data.loc[test_mask, [c for c in data.columns if c != 'target']]
    y_test = data.loc[test_mask, ['target']]
    X_test.to_csv('X_test.csv')

    if is_train:
        clf = train(X_train, y_train, prefix)
        with open('ml/model/{}_clf_binary.pkl'.format(prefix), 'wb') as f:
            pickle.dump(clf, f)
    else:
        #with open('ml/model/{}_clf_binary.pkl'.format(prefix), 'rb') as f:
        with open('ml/model/201901010000_201902021759_clf_binary.pkl', 'rb') as f:
            clf = pickle.load(f)

    y_label = clf.predict(X_train)
    y_pred = clf.predict_proba(X_train)[:, 1]
    auc = metrics.roc_auc_score(y_train, y_pred)
    confusion_mat = confusion_matrix(y_train, y_label)
    tp, fn, fp, tn = confusion_mat.ravel()
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    logger.debug('Train: (auc, accuracy, precision, recall) = ({}, {}, {}, {})'.format(auc, accuracy, precision, recall))

    norm_mean = y_pred.mean()
    norm_std = y_pred.std()

    y_label = clf.predict(X_test)
    y_pred = clf.predict_proba(X_test)[:, 1]
    auc = metrics.roc_auc_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_label)
    tp, fn, fp, tn = confusion_mat.ravel()
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    logger.debug('Test: (auc, accuracy, precision, recall) = ({}, {}, {}, {})'.format(auc, accuracy, precision, recall))
    X_test[[c for c in X_test.columns if '1min' in c]].to_csv('X_test_5min.csv')
    X_test[[c for c in X_test.columns if '5min' in c]].to_csv('X_test_5min.csv')
    print(pd.Series(y_pred[:-1], index=X_test.index[1:]))

    # 投資指標（予測値の確率）とリターンの関係

    #norm_mean = 0.4981990534953142
    #norm_std = 0.28677379837073347
    z_scores = (y_pred - norm_mean) / norm_std
    logger.debug('mean: {}, std: {}'.format(norm_mean, norm_std))

    # z_scoreと実際の利益の相関係数
    #spread = 0.062
    spread = 0
    # 注意: buy -> 相手が買う値段なので自分が売る時の値段、sell -> 相手が売る値段なので自分が買う時の値段
    prices = pd.DataFrame({'buy': ((100 - spread) / (100 + spread)) * X_test['close_1min'], 'sell': X_test['close_1min']})
    profits = -X_test['close_1min'].diff(-1)
    profits.iloc[-1] = 0  # ズレた場所を補間する
    logger.debug(np.corrcoef(z_scores, profits))

    #pd.concat([pd.concat([profits, y_test], axis=1), pd.Series(y_pred, index=y_test.index)], axis=1).to_csv('test.csv')

    # 資産推移を取得
    dts = prices.index
    import itertools
    threshold_uppers = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    threshold_lowers = [-0.1, -0.3, -0.5, -0.7, -0.9, -1.1, -1.3, -1.5]

    for threshold_upper, threshold_lower in itertools.product(threshold_uppers, threshold_lowers):
        # 計算した指標は実際の運用では1分前のものなので１分ずらす
        assets = get_assets_change(z_scores[:-1], prices[1:], dts[1:], threshold_upper, threshold_lower)
        print('{} {} {}'.format(threshold_upper, threshold_lower, assets[-1]))

        # 可視化
        fig, ax = plt.subplots(3, 1, figsize=(12, 18))
        ax[0].scatter(z_scores, profits, alpha=0.3)
        ax[0].set_title('predict probability vs profit after 1min')
        ax[0].set_xlabel('predict probability')
        ax[0].set_ylabel('profit after 1min')
        ax[1].hist(z_scores)
        ax[1].set_title('predict probability')
        ax[2].plot(dts[1:], assets)
        ax[2].set_title('assets')
        #plt.show()

    pd.Series(y_pred).to_csv('y_pred.csv')
    X_test.to_csv('X_test.csv')

if __name__=='__main__':
    main()
