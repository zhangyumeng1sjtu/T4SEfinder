import sys
from Bio import SeqIO

fasta = sys.argv[1]
filtered_records = []

for record in SeqIO.parse(fasta, "fasta"):
	if len(record) >= 50 and len(record) <= 5000 and "pseudo=true" not in record.description and 'X' not in str(record.seq):
	    filtered_records.append(record)

SeqIO.write(filtered_records, fasta, "fasta")
