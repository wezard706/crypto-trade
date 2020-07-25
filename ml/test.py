import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn import metrics

from common.utils import get_dt_format
from ml.create_dataset import create_dataset_api
from ml.train_binary import extract_feature, create_target, get_assets_change

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
# ファイル出力(debug)
dfile_handler = logging.FileHandler('ml/log/debug.log', 'a')
dfile_handler.setLevel(logging.DEBUG)
dfile_handler.setFormatter(format)
logger.addHandler(dfile_handler)
# ファイル出力(error)
efile_handler = logging.FileHandler('ml/log/error.log', 'a')
efile_handler.setLevel(logging.ERROR)
efile_handler.setFormatter(format)
logger.addHandler(efile_handler)

def main():
    candle_types = ['1min', '5min']
    start_dt = '2019-01-21 00:00:00'
    end_dt = '2019-01-23 23:59:59'
    model_path = 'ml/model/2018011302055_201901232359_clf_binary.pkl'
    norm_mean = 0.31429959221078313
    norm_std = 0.2268023096336549
    threshold = 0.1

    # データを取得
    start_dt_create_data = (datetime.strptime(start_dt, get_dt_format()) - timedelta(minutes=185)).strftime(get_dt_format())
    candles_all = {}
    for candle_type in candle_types:
        candles_all[candle_type] = create_dataset_api(start_dt_create_data, end_dt, pair='xrp_jpy', candle_type=candle_type)

    # 特徴量抽出
    features = extract_feature(start_dt, end_dt, 'xrp_jpy', candle_types, df_all=candles_all)
    target = create_target(candles_all['1min'], start_dt, end_dt)
    data = pd.concat([features, target], axis=1)
    data.columns = features.columns.tolist() + ['target']
    data = data.loc[~data.target.isnull()]
    X_test = data.loc[:, [c for c in data.columns if c != 'target']]
    y_test = data.loc[:, ['target']]

    # 予測
    with open(model_path, mode='rb') as f:
        clf = pickle.load(f)
    y_label = clf.predict(X_test)
    y_pred = clf.predict_proba(X_test)[:, 1]
    z_scores = (y_pred - norm_mean) / norm_std
    logger.debug(metrics.confusion_matrix(y_test, y_label))
    logger.debug('Test AUC: %.6f' % (metrics.roc_auc_score(y_test, y_pred)))

    # z_scoreと実際の利益の相関係数
    prices = X_test['close_1min']
    profits = -prices.diff(-1)
    profits.iloc[-1] = 0  # ズレた場所を補間する
    logger.debug(np.corrcoef(z_scores, profits))

    # 資産推移を取得
    dts = prices.index
    assets = get_assets_change(z_scores, prices, dts, threshold)

    # 可視化
    fig, ax = plt.subplots(3, 1, figsize=(12, 18))
    ax[0].scatter(z_scores, profits, alpha=0.3)
    ax[0].set_title('predict probability vs profit after 1min')
    ax[0].set_xlabel('predict probability')
    ax[0].set_ylabel('profit after 1min')
    ax[1].hist(z_scores)
    ax[1].set_title('predict probability')
    ax[2].plot(dts, assets)
    ax[2].set_title('assets')
    plt.show()

    pd.Series(y_pred).to_csv('y_pred.csv')
    X_test.to_csv('X_test.csv')

if __name__=='__main__':
    main()