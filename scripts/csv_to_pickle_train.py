import pandas as pd


# %%
dfile = "/home/nfs/kkaminski/kaggle/train_data.csv"

reactivity_cols = [f'reactivity_{n:04}' for n in range(1, 206+1)]
data = pd.read_csv(dfile, engine='c', usecols=['sequence_id', 'experiment_type', 'SN_filter', 'sequence'] + reactivity_cols)
data_snfilter = data[data.SN_filter == 1].copy()
data_snfilter.to_pickle(dfile.replace('.csv', '.snfilter.p.gz'), compression='gzip')
data.to_pickle(dfile.replace('.csv', '.p.gz'), compression='gzip')
print('done')