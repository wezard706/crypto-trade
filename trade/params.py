def get_params():
    params = {
        # 基本設定
        'pair': 'xrp_jpy',
        'candle_type': '1min',
        'amount': 0.1,
        'order_type': 'limit',
        'asset_lower': 1000,

        # ModelLgb
        'norm_mean': 0.49390813464655126,
        'norm_std': 0.29723635781506785,
        'lower': -1.0,
        'upper': 1.0,

        # ModelNa
        'reward_upper': 0.06,
        'loss_lower': -0.03,
    }
    return params