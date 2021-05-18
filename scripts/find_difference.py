import sys
import os

from Bio import SeqIO

sequences = sys.argv[1]
pssm_dir = sys.argv[2]

seq_dict = SeqIO.to_dict(SeqIO.parse(sequences, "fasta"))
seqids = list(seq_dict.keys())
pssms = [os.path.splitext(pssm)[0] for pssm in os.listdir(pssm_dir)]

difference = list(set(seqids).difference(set(pssms)))

diff_records = []
for name in difference:
	diff_records.append(seq_dict[name])
if len(diff_records) > 0:
	SeqIO.write(diff_records,f"{sequences}.par","fasta")
    