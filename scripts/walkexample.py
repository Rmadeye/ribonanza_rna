import os

basedir = "/home/nfs/kkaminski/kaggle/Ribonanza_bpp_files/extra_data"

for itr, (dirpath, dir, files) in enumerate(os.walk(basedir)):
    print(dirpath, dir, files)