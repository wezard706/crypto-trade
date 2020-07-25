import os
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

from ml.params import get_params
from common.utils import calc_rmse, plot_confusion_matrix

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

def train(X_train, y_train, args):
    '''
    params_grid = {
        'num_leaves': [8, 32, 128, 1024], # モデルの表現力
        'min_data_in_leaf': [10, 100, 1000], # 過学習を調整
        'n_estimators': [100], # 高いと過学習する、learning_rateを下げて調整
        'learning_rate': [0.1] # 低いほどロバスト、n_estimatorsと反比例
    }

    # rmse: mean 0.0017372877337038517, std 0.000755704939365387, test rmse: 0.001726324437186122
    params_grid = {
        'num_leaves': [50],  # モデルの表現力
        'min_data_in_leaf': [80],  # 過学習を調整
        'n_estimators': [100],  # 高いと過学習する、learning_rateを下げて調整
        'learning_rate': [0.1]  # 低いほどロバスト、n_estimatorsと反比例
    }
    '''

    params_grid = {
        'num_leaves': [45, 50, 55],  # モデルの表現力
        'min_data_in_leaf': [70, 75, 80, 85, 90],  # 過学習を調整
        'n_estimators': [100],  # 高いと過学習する、learning_rateを下げて調整
        'learning_rate': [0.1]  # 低いほどロバスト、n_estimatorsと反比例
    }

    best_rmse = 99999
    best_rmse_std = None
    best_imp = None
    best_imp_std = None
    best_param = None
    logger.debug('grid search')
    for params in itertools.product(*params_grid.values()):
        param = {k: v for k, v in zip(params_grid.keys(), params)}

        list_rmse = []
        list_imp = []
        start_train = X_train.index[0]
        end_train = start_train + timedelta(days=args['cv_range_train'])
        start_valid = end_train
        end_valid = start_valid + timedelta(days=args['cv_range_valid'])
        while end_valid <= X_train.index[-1]:
            logger.debug('train: {} - {}, valid: {} - {}'.format(start_train, end_train, start_valid, end_valid))
            train_mask = (start_train <= X_train.index) & (X_train.index < end_train)
            valid_mask = (start_valid <= X_train.index) & (X_train.index < end_valid)
            _X_train = X_train[train_mask]
            _y_train = y_train[train_mask]
            _X_valid = X_train[valid_mask]
            _y_valid = y_train[valid_mask]

            clf = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression', class_weight="balanced", nthread=4, **param)
            clf.fit(_X_train.values, _y_train.values)
            pred_valid = clf.predict(_X_valid)
            list_rmse.append(calc_rmse(_y_valid, pred_valid))
            list_imp.append(clf.feature_importances_)

            #logger.debug('f1: mean {}, std {} current {}'.format(np.mean(list_f1), np.std(list_f1), list_f1[-1]))
            del _X_train, _X_valid, _y_train, _y_valid
            gc.collect()

            start_train += timedelta(days=args['stride'])
            end_train = start_train + timedelta(days=args['cv_range_train'])
            start_valid = end_train
            end_valid = start_valid + timedelta(days=args['cv_range_valid'])

        mean_rmse = np.mean(list_rmse)
        std_rmse = np.std(list_rmse)
        imp_mean = np.mean(list_imp, axis=0)
        imp_std = np.std(list_imp, axis=0)
        logger.debug('{}, rmse: mean {}, std {}'.format(param, mean_rmse, std_rmse))
        if best_rmse > mean_rmse:
            best_rmse = mean_rmse
            best_rmse_std = std_rmse
            best_imp = imp_mean
            best_imp_std = imp_std
            best_param = param

    logger.debug('best parameter: {} rmse: {} std: {}'.format(best_param, best_rmse, best_rmse_std))

    imp = np.concatenate([best_imp.reshape(len(best_imp), 1), best_imp_std.reshape(len(best_imp_std), 1)], axis=1)
    imp = pd.DataFrame(imp, columns=['mean', 'std'], index=X_train.columns)
    imp = imp.sort_values('mean', ascending=False)
    imp.to_csv('ml/result/importance.csv')
    logger.debug('importance: \n {}'.format(imp))

    # 四分位点を求めるため、上限・下限ともに25%を2で等分した0.125、0.875の位置で予測する
    alpha = 0.875
    # model 定義
    clf_upper = lgb.LGBMRegressor(boosting_type='gbdt', objective='quantile', class_weight="balanced", alpha=alpha, nthread=4, **best_param)
    clf_upper.fit(X_train.values, y_train.values)

    # alphaを反転して下限の予測
    clf_lower = lgb.LGBMRegressor(boosting_type='gbdt', objective='quantile', class_weight="balanced", alpha=1.0 - alpha, nthread=4, **best_param)
    clf_lower.fit(X_train.values, y_train.values)

    # 損失関数を最小2乗法に設定して予測
    clf = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression', class_weight="balanced", nthread=4, **best_param)
    clf.fit(X_train.values, y_train.values)
    return [clf, clf_upper, clf_lower]

def calc_change(line1, line2):
    diff = line1 - line2
    change = pd.Series(np.where(diff >= 0, 0.5, -0.5), index=diff.index).diff() # 2階微分（変化点）
    return change

def _extract_feature(candle, params):
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

    # RSI
    features['rsi_s'] = ta.RSI(c, timeperiod=params['rsi_timeperiod_s'])
    features['rsi_m'] = ta.RSI(c, timeperiod=params['rsi_timeperiod_m'])
    features['rsi_l'] = ta.RSI(c, timeperiod=params['rsi_timeperiod_l'])

    # STOCHF
    fastk, fastd = ta.STOCHF(h, l, c, fastk_period=params['stockf_fastk_period_s'], fastd_period=params['stockf_fastd_period_s'], fastd_matype=ta.MA_Type.T3)
    change_stockf = calc_change(fastk, fastd)
    change_stockf.index = fastk.index
    features['fastk_s'] = fastk
    features['fastd_s'] = fastd
    features['fast_change_s'] = change_stockf
    fastk, fastd = ta.STOCHF(h, l, c, fastk_period=params['stockf_fastk_period_m'], fastd_period=params['stockf_fastd_period_m'], fastd_matype=ta.MA_Type.T3)
    change_stockf = calc_change(fastk, fastd)
    change_stockf.index = fastk.index
    features['fastk_m'] = fastk
    features['fastd_m'] = fastd
    features['fast_change_m'] = change_stockf
    fastk, fastd = ta.STOCHF(h, l, c, fastk_period=params['stockf_fastk_period_l'], fastd_period=params['stockf_fastd_period_l'], fastd_matype=ta.MA_Type.T3)
    change_stockf = calc_change(fastk, fastd)
    change_stockf.index = fastk.index
    features['fastk_l'] = fastk
    features['fastd_l'] = fastd
    features['fast_change_l'] = change_stockf

    # WILLR
    features['willr_s'] = ta.WILLR(h, l, c, timeperiod=params['willr_timeperiod_s'])
    features['willr_m'] = ta.WILLR(h, l, c, timeperiod=params['willr_timeperiod_m'])
    features['willr_l'] = ta.WILLR(h, l, c, timeperiod=params['willr_timeperiod_l'])

    # MACD
    macd, macdsignal, macdhist = ta.MACDEXT(c, fastperiod=params['macd_fastperiod_s'],
                                            slowperiod=params['macd_slowperiod_s'],
                                            signalperiod=params['macd_signalperiod_s'],
                                            fastmatype=ta.MA_Type.T3, slowmatype=ta.MA_Type.T3,
                                            signalmatype=ta.MA_Type.T3)
    change_macd = calc_change(macd, macdsignal)
    change_macd.index = macd.index
    features['macd_s'] = macd
    features['macdsignal_s'] = macdsignal
    features['macdhist_s'] = macdhist
    features['change_macd_s'] = change_macd
    macd, macdsignal, macdhist = ta.MACDEXT(c, fastperiod=params['macd_fastperiod_m'],
                                            slowperiod=params['macd_slowperiod_m'],
                                            signalperiod=params['macd_signalperiod_m'],
                                            fastmatype=ta.MA_Type.T3, slowmatype=ta.MA_Type.T3,
                                            signalmatype=ta.MA_Type.T3)
    change_macd = calc_change(macd, macdsignal)
    change_macd.index = macd.index
    features['macd_m'] = macd
    features['macdsignal_m'] = macdsignal
    features['macdhist_m'] = macdhist
    features['change_macd_m'] = change_macd

    # ROCP
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

    # ボリンジャーバンド
    bb_upper, bb_middle, bb_lower = ta.BBANDS(c, timeperiod=params['bbands_timeperiod_s'],
                                              nbdevup=params['bbands_nbdevup_s'], nbdevdn=params['bbands_nbdevdn_s'],
                                              matype=ta.MA_Type.T3)
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
                                              matype=ta.MA_Type.T3)
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
                                              matype=ta.MA_Type.T3)
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

    # 出来高
    features['obv'] = ta.OBV(c, v)
    features['ad'] = ta.AD(h, l, c, v)
    features['adosc_s'] = ta.ADOSC(h, l, c, v, fastperiod=params['fastperiod_adosc_s'],
                                   slowperiod=params['slowperiod_adosc_s'])

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
    features['CDLDOJI'] = ta.CDLDOJI(o, h, l, c)
    features['CDLDOJISTAR'] = ta.CDLDOJISTAR(o, h, l, c)
    features['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(o, h, l, c)
    features['CDLENGULFING'] = ta.CDLENGULFING(o, h, l, c)
    features['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(o, h, l, c, penetration=0)
    features['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(o, h, l, c, penetration=0)
    features['CDLGAPSIDESIDEWHITE'] = ta.CDLGAPSIDESIDEWHITE(o, h, l, c)
    features['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(o, h, l, c)
    features['CDLHAMMER'] = ta.CDLHAMMER(o, h, l, c)
    features['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(o, h, l, c)
    features['CDLHARAMI'] = ta.CDLHARAMI(o, h, l, c)
    features['CDLHARAMICROSS'] = ta.CDLHARAMICROSS(o, h, l, c)
    features['CDLHIGHWAVE'] = ta.CDLHIGHWAVE(o, h, l, c)
    features['CDLHIKKAKE'] = ta.CDLHIKKAKE(o, h, l, c)
    features['CDLHIKKAKEMOD'] = ta.CDLHIKKAKEMOD(o, h, l, c)
    features['CDLHOMINGPIGEON'] = ta.CDLHOMINGPIGEON(o, h, l, c)
    features['CDLIDENTICAL3CROWS'] = ta.CDLIDENTICAL3CROWS(o, h, l, c)
    features['CDLINNECK'] = ta.CDLINNECK(o, h, l, c)
    features['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(o, h, l, c)
    features['CDLKICKING'] = ta.CDLKICKING(o, h, l, c)
    features['CDLKICKINGBYLENGTH'] = ta.CDLKICKINGBYLENGTH(o, h, l, c)
    features['CDLLADDERBOTTOM'] = ta.CDLLADDERBOTTOM(o, h, l, c)
    features['CDLLONGLEGGEDDOJI'] = ta.CDLLONGLEGGEDDOJI(o, h, l, c)
    features['CDLMARUBOZU'] = ta.CDLMARUBOZU(o, h, l, c)
    features['CDLMATCHINGLOW'] = ta.CDLMATCHINGLOW(o, h, l, c)
    features['CDLMATHOLD'] = ta.CDLMATHOLD(o, h, l, c, penetration=0)
    features['CDLMORNINGDOJISTAR'] = ta.CDLMORNINGDOJISTAR(o, h, l, c, penetration=0)
    features['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(o, h, l, c, penetration=0)
    features['CDLONNECK'] = ta.CDLONNECK(o, h, l, c)
    features['CDLPIERCING'] = ta.CDLPIERCING(o, h, l, c)
    features['CDLRICKSHAWMAN'] = ta.CDLRICKSHAWMAN(o, h, l, c)
    features['CDLRISEFALL3METHODS'] = ta.CDLRISEFALL3METHODS(o, h, l, c)
    features['CDLSEPARATINGLINES'] = ta.CDLSEPARATINGLINES(o, h, l, c)
    features['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(o, h, l, c)
    features['CDLSHORTLINE'] = ta.CDLSHORTLINE(o, h, l, c)
    features['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(o, h, l, c)
    features['CDLSTALLEDPATTERN'] = ta.CDLSTALLEDPATTERN(o, h, l, c)
    features['CDLSTICKSANDWICH'] = ta.CDLSTICKSANDWICH(o, h, l, c)
    features['CDLTAKURI'] = ta.CDLTAKURI(o, h, l, c)
    features['CDLTASUKIGAP'] = ta.CDLTASUKIGAP(o, h, l, c)
    features['CDLTHRUSTING'] = ta.CDLTHRUSTING(o, h, l, c)
    features['CDLTRISTAR'] = ta.CDLTRISTAR(o, h, l, c)
    features['CDLUNIQUE3RIVER'] = ta.CDLUNIQUE3RIVER(o, h, l, c)
    features['CDLUPSIDEGAP2CROWS'] = ta.CDLUPSIDEGAP2CROWS(o, h, l, c)
    features['CDLXSIDEGAP3METHODS'] = ta.CDLXSIDEGAP3METHODS(o, h, l, c)

    window = 5
    features_ext = features
    for w in range(window):
        tmp = features.shift(periods=60 * (w + 1), freq='S')
        tmp.columns = [c + '_' + str(w + 1) + 'w' for c in features.columns]
        features_ext = pd.concat([features_ext, tmp], axis=1)
    return features_ext

def format_candles(candles):
    candles_formatted = {}
    candle_types = ['1min', '5min', '15min', '30min', '1hour', '4hour', '8hour', '12hour', '1day', '1week', '1month']
    for candle_type in candle_types:
        tmp = candles.loc[:, [s + '_' + candle_type for s in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]]
        tmp.columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        candles_formatted[candle_type] = tmp
    return candles_formatted

def summary_candle(df):
    return df.groupby('timestamp').agg({'open': lambda x: x.iloc[0], 'high': max, 'low': min, 'close': lambda x: x.iloc[-1], 'volume': lambda x: x.iloc[-1], 'timestamp': lambda x: x.iloc[-1]})

def extract_feature(df, start_dt, end_dt, candle_types):
    candles = format_candles(df)

    features = None
    for candle_type in candle_types:
        logger.debug('extract features for {}'.format(candle_type))

        candle = summary_candle(candles[candle_type])
        candle.index = candle.index.map(lambda x: datetime.fromtimestamp(x / 1000))
        candle = candle.loc[(candle.index > start_dt) & (candle.index < end_dt)]

        params = get_params('1min')
        feature = _extract_feature(candle, params)
        feature = feature.fillna(method='ffill')
        feature.columns = [c + '_' + candle_type for c in feature.columns]

        if features is None:
            features = feature
        else:
            features = pd.concat([features, feature], axis=1)
            features = features.fillna(method='ffill')
    return features

def create_target(df, start_dt, end_dt, candle_type):
    candles = format_candles(df)
    candle = summary_candle(candles[candle_type])
    candle.index = candle.index.map(lambda x: datetime.fromtimestamp(x / 1000))

    rocp = ta.ROCP(ta.EMA(candle.close, timeperiod=5), timeperiod=5)
    rocp_next = rocp.shift(periods=-300, freq='S')
    target = rocp_next.loc[(rocp_next.index > start_dt) & (rocp_next.index < end_dt)]
    return target

def main():
    is_feature_extraction = True
    is_train = True

    args = {
        'start_dt': '2018-11-19 00:00:00',
        'end_dt': '2019-01-13 23:59:59',
        'train_test_split': '2019-01-01 00:00:00',
        'cv_range_train': 21,
        'cv_range_valid': 1,
        'stride': 1
    }

    start_dt = datetime.strptime(args['start_dt'], '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(args['end_dt'], '%Y-%m-%d %H:%M:%S')

    with open('ml/input/20181119_20190113_candles.pkl', mode='rb') as f:
        df = pickle.load(f)

    if is_feature_extraction:
        features = extract_feature(df, start_dt, end_dt, candle_types=['1min', '5min', '15min', '30min'])
        features.to_pickle('ml/input/features.pkl')
    else:
        with open('ml/input/features.pkl', mode='rb') as f:
            features = pickle.load(f)

    target = create_target(df, start_dt, end_dt, candle_type='1min')

    # targetが欠損していない特徴量のみ使用する
    data = pd.concat([features, target], axis=1)
    data.columns = features.columns.tolist() + ['target']
    data = data.loc[~data.target.isnull()]

    train_mask = args['train_test_split'] > data.index
    test_mask = args['train_test_split'] <= data.index
    X_train = data.loc[train_mask, [c for c in data.columns if c != 'target']]
    y_train = data.loc[train_mask, ['target']]
    X_test = data.loc[test_mask, [c for c in data.columns if c != 'target']]
    y_test = data.loc[test_mask, ['target']]

    if is_train:
        clf, clf_upper, clf_lower = train(X_train, y_train, args)
        with open('ml/model/20181119_20190113_clf.pkl', 'wb') as f:
            pickle.dump(clf, f)
        with open('ml/model/20181119_20190113_clf_upper.pkl', 'wb') as f:
            pickle.dump(clf_upper, f)
        with open('ml/model/20181119_20190113_clf_lower.pkl', 'wb') as f:
            pickle.dump(clf_lower, f)
    else:
        with open('ml/model/20181119_20190113_clf.pkl', mode='rb') as f:
            clf = pickle.load(f)
        with open('ml/model/20181119_20190113_clf_upper.pkl', mode='rb') as f:
            clf_upper = pickle.load(f)
        with open('ml/model/20181119_20190113_clf_lower.pkl', mode='rb') as f:
            clf_lower = pickle.load(f)
            
if __name__=='__main__':
    main()