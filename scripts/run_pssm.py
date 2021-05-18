# -*- coding: utf-8 -*-

'''
Microbial Bioinformatics Group in MML,SJTU
run_pssm.py: Script for generating PSSM profiles
Yumeng Zhang <zhangyumeng1@sjtu.edu.cn>

Usage: 
    $ python run_pssm.py -i <input sequence>
                         -db <blast database>
                         -e <evalue>
                         -n <num_iteration> 
                         -o <ouput directory>
                         -threads <num_threads>
'''

import os
import argparse

from Bio import SeqIO
import tempfile


def pssm_command(query, db, evalue, n_iter, out_file, pssm_file, num_threads):
    """
    Single PSI-BLAST command
    @param query: single query protein sequence
    @param db: database used for PSI-BLAST
    @param evalue: BLAST evalue
    @param n_iter: num. of PSI-BLAST iterations
    @param out_file: BLAST output file
    @param pssm_file: output .pssm file
    @param num_threads: threads used for PSSM-BLAST 
    """   
    print("Generating PSSM profile for sequence %s ..." % os.path.basename(query)) 
    os.system(f'psiblast \
                -query "{query}" \
                -db {db} \
                -num_iterations {n_iter} \
                -evalue {evalue} \
                -out "{out_file}" \
                -out_ascii_pssm "{pssm_file}" \
                -num_threads {num_threads} \
                2>/dev/null')


def pssm(sequences, db, evalue, n_iter, outdir, num_threads):
    """
    Obtain all .pssm files for query protein sequences
    @param sequences,: query protein sequences
    @param db: database used for PSI-BLAST
    @param evalue: BLAST evalue
    @param n_iter: num. of PSI-BLAST iterations
    @param outdir: directory for output of PSI-BLAST
    @param num_threads: threads used for PSI-BLAST 
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    temp_dir = tempfile.TemporaryDirectory()

    for record in SeqIO.parse(sequences, "fasta"):
        query = os.path.join(temp_dir.name, "%s.fasta" % record.id)
        SeqIO.write([record], query, "fasta")
        output_file = os.path.join(temp_dir.name,"%s.out" % record.id)
        pssm_file = os.path.join(outdir,"%s.pssm" % record.id)
        pssm_command(query, db, evalue, n_iter, output_file, pssm_file, num_threads)
    
    temp_dir.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Generate PSSM files")

    parser.add_argument('-i', '--input_sequence', type=str, required=True)
    parser.add_argument('-db', '--blast_database', type=str, required=True)
    parser.add_argument('-e', '--evalue', type=float, default=1e-3)
    parser.add_argument('-n', '--num_iterations', type=int, default=3)
    parser.add_argument('-o', '--output_dir', type=str, default="pssm_files")
    parser.add_argument('-threads', '--num_threads', type=int, default=1)

    args = parser.parse_args()

    pssm(args.input_sequence, args.blast_database, args.evalue, args.num_iterations, args.output_dir, args.num_threads)


if __name__ == "__main__":
    main()
