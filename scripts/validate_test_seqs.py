import torch
import pandas as pd


rna_dict = dict(A=0, C=1, G=2, U=3)
rna_map = lambda letter : rna_dict[letter]



print('test tensor loaded')
testdf = pd.read_csv("/home/nfs/kkaminski/kaggle/test_sequences.csv", engine="c", nrows=1000)

seqs=testdf['sequence'].tolist()
sequences = [list(map(rna_map, seq)) for seq in seqs]
test_t = torch.load("/home/nfs/rmadaj/ai/kaggle/kaggle_rna/data/inputs/kaggle_test_sequences.pt")

print(test_t[0])
print(sequences[0])