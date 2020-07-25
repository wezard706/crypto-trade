import pandas as pd

start_dt = '2019-02-02 12:00:00'
end_dt = '2019-02-02 12:59:59'
#columns = ['open_1min', 'high_1min', 'low_1min', 'close_1min', 'volume_1min', 'open_5min', 'high_5min', 'low_5min', 'close_5min', 'volume_5min']

# real
df_real = None
for i in range(0, 60):
    tmp = pd.read_csv('X_2019-02-02 12:{0:02d}:00.csv'.format(i))
    if df_real is None:
        df_real = tmp
    else:
        df_real = pd.concat([df_real, tmp])
df_real.index = pd.to_datetime(df_real.timestamp)
columns = [col for col in df_real.columns if col != 'timestamp']
df_real = df_real[columns]

# sim
df_sim = pd.read_csv('X_test.csv')
df_sim.index = pd.to_datetime(df_sim.timestamp)
df_sim = df_sim[(start_dt <= df_sim.index) & (df_sim.index <= end_dt)]
df_sim = df_sim[columns]

# diff
df_diff = pd.DataFrame()
for col in columns:
    df_diff[col] = df_sim[col] - df_real[col]
df = pd.concat([df_sim, df_real, df_diff], axis=1)

# save
start_dt_str = start_dt[:4] + start_dt[5:7] + start_dt[8:10] + start_dt[11:13] + start_dt[14:16]
end_dt_str = end_dt[:4] + end_dt[5:7] + end_dt[8:10] + end_dt[11:13] + end_dt[14:16]
df.to_csv('{}_{}_real_sim.csv'.format(start_dt_str, end_dt_str))