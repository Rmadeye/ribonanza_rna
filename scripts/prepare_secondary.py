
import os
import subprocess
import argparse
import tempfile

from Bio import SeqIO
from tqdm import tqdm
import pandas as pd

def get_parser():

    parser = argparse.ArgumentParser('predict SS for given RNA sequences from csv file')
    parser.add_argument('input', help='input csv file')
    parser.add_argument('output', help='output csv file')

    args = parser.parse_args()
    return args

def seqlist_to_fasta_sting(ids, seqlist) -> str:
    """
    return file name
    """
    string = ""
    for idx, seq in zip(ids, seqlist):
        string += f"> {idx}\n"
        string += f"{seq}\n"
    return string


def main(args):
    COLUMNS = ['description', 'secondary']
    CHUNKSIZE=100
    RNAFOLD_BIN = "/home/users/kkaminski/kaggle/rnafold/ViennaRNA-2.6.4/src/bin/RNAfold"
    CHUNKOUT = "/home/nfs/kkaminski/tmp/rnafold_ss.fasta"
    CHUNKIN = "/home/nfs/kkaminski/tmp/rnafold_seq.fasta"
    assert os.path.isfile(args.input)
    assert os.path.isfile(RNAFOLD_BIN)
    print('input', args.input)
    print('output: ', args.output)
    df_iterator = pd.read_csv(args.input, chunksize=CHUNKSIZE)
    print()
    for i, chunk in tqdm(enumerate(df_iterator)):
        sequences = chunk.sequence.tolist()
        ids = list(range(len(sequences)))
        with open(CHUNKIN, mode="wt") as fp:
            string = seqlist_to_fasta_sting(ids, sequences)
            fp.write(string)

        proc = subprocess.run([RNAFOLD_BIN, "-T" ,"37", "--noPS", CHUNKIN], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        assert proc.stderr is None or len(proc.stderr) == 0, f"{proc.stdout}: {proc.stderr}"
        with open(CHUNKOUT, 'wt') as fp:
            fp.write(proc.stdout.decode("utf-8"))
        datass = SeqIO.parse(CHUNKOUT, 'fasta')
        # unpack
        datass = [[record.description, str(record.seq)] for i, record in enumerate(datass)]
        breakpoint()
        data = pd.DataFrame(datass, columns=COLUMNS)
        data['id'] = ids
        data['sequence'] = sequences
        mode = 'w' if i == 0 else 'a'
        header = i == 0
        data.to_csv(args.output, mode=mode, header=header, index=False)
        


if __name__ == "__main__":
    args = get_parser()
    main(args)
