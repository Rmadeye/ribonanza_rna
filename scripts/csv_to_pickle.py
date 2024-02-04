import pandas as pd


# %%
dfile = "/home/nfs/kkaminski/kaggle/train_data.csv"
dfile_test = "/home/nfs/kkaminski/kaggle/test_sequences.csv"
dfile_sec =  "/home/nfs/kkaminski/kaggle/train_secondary.csv"
dfile_sec_test = "/home/nfs/kkaminski/kaggle/test_secondary.csv"
basedir = "/home/nfs/kkaminski/kaggle/Ribonanza_bpp_files/extra_data"
outdir = "/home/nfs/kkaminski/kaggle/structure"

data = pd.read_csv(dfile, engine='c', usecols=['sequence_id', 'experiment_type', 'SN_filter', 'sequence'])
data.to_pickle(dfile.replace('.csv', '.p.gz'), compression='gzip')

restcsvs = [dfile_test, dfile_sec, dfile_sec_test]
for f in restcsvs:
    tmp = pd.read_csv(f)
    if f.find('secondary') != -1:
        tmp['seqlen'] = tmp['sequence'].apply(len)
        tmp.drop_duplicates(['sequence'], inplace=True)
        # fix bad saving error
        tmp['secondary'] = tmp.apply(lambda row: row['secondary'][row['seqlen']: 2*row['seqlen']], axis=1)
    tmp.to_pickle(f.replace('.csv', '.p.gz'), compression='gzip')

print('done')