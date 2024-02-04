import pandas as pd

num_records = 269796671
columns = ['id', 'reactivity_DMS_MaP', 'reactivity_2A3_MaP']
data = '/home/nfs/rmadaj/ai/kaggle/kaggle_rna/submission.csv'

d = pd.read_csv(data, nrows=10)
for col in columns:
    assert col in d.columns, f'missing {col} in submission file {data}'
d = pd.read_csv(data, engine='c')
assert d.shape[0] == num_records, "submission file has wrong number of records"