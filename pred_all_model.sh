#!/bin/bash
# usage: ./pred_all_model.sh Acession_Number (e.g. NC_009494)

acc=$1
res=results/$acc
fasta=$res/sequence.fasta
pssmdir=$res/pssm_files
####################################################
mkdir -p $res
echo "Downloading protein sequences corresponding to Accession number: "$acc" ..."
efetch -db nuccore -id $acc -format fasta_cds_aa > $fasta
python scripts/filter.py $fasta
####################################################
echo "Running PSI-BLAST to generate PSSM profiles for "$fasta" ..."
python scripts/run_pssm.py -i $fasta -db blastdb/swissprot -e 10 -n 3 -o $pssmdir -threads 16
python scripts/find_difference.py $fasta $pssmdir
if [ -f $fasta.par ]; then
    python scripts/run_pssm.py -i $fasta.par -db blastdb/swissprot -e 50 -n 3 -o $pssmdir -threads 16
fi
####################################################
echo "Predicting T4SEs in "$fasta" ..."
python main.py -in $fasta \
               -weights weights/mlp/ \
               -out $res/tapebert_mlp \
               --vote_required \
               tapebert_mlp \

python main.py -in $fasta \
               -weights weights/svm/ \
               -out $res/tapebert_svm  \
               --vote_required \
               tapebert_svm \

python main.py -in $fasta \
               -weights weights/cnn/ \
               -out $res/pssm_cnn \
               --vote_required \
               pssm_cnn \
               -pssm $pssmdir

python main.py -in $fasta \
               -weights weights/bilstm/ \
               -out $res/hybrid_bilstm  \
               --vote_required \
               hybrid_bilstm \
               -pssm $pssmdir
####################################################
echo "You can view the summarized results in "$res"/summary.txt"  
python scripts/summary.py -i $res
####################################################