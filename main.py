# -*- coding: utf-8 -*-

'''
Microbial Bioinformatics Group in MML,SJTU
main.py: Main script for T4SEs prediction.
Yumeng Zhang <zhangyumeng1@sjtu.edu.cn>

Usage:
    $ python main.py -in <input sequence>
                     -weights <weight directory>
                     -seed <random seed>
                     -out <output directory>
                     -vote [-threshold <vote threshold>]
                     {tapebert_mlp,tapebert_svm,pssm_cnn,hybrid_bilstm}
                     [-pretrain <pre-trained model>]
                     [-pssm <pssmfile directory>]

'''

import argparse
import os

import torch
import joblib

from T4SEfinder.model import *
from T4SEfinder.prediction import *
from T4SEfinder.dataset import *
from T4SEfinder.utils import set_seed


def handle_tapebert_mlp(args):
    set_seed(args.random_seed)
    embed_path = f'{os.path.splitext(args.input_sequence)[0]}.bert.npz'
    if not os.path.exists(embed_path):
        BertEmbedding(args.input_sequence, embed_path, args.pretrained_model, args.random_seed)
    if not os.path.exists(args.output_dir) and args.output_dir != "":
        os.makedirs(args.output_dir)
    model = SimpleMLP()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for i, weights in enumerate(os.listdir(args.weights_dir)):
        model.load_state_dict(torch.load(os.path.join(args.weights_dir, weights), map_location=device))
        dataset = TestDataset(args.input_sequence, embed_path, feature="avg")
        predict(model, dataset, os.path.join(args.output_dir, f'pred_result_{i+1}.txt'), device)
    if args.vote_required:
        vote_predict(args.input_sequence, args.output_dir, args.vote_threshold or 0.5)


def handle_tapebert_svm(args):
    set_seed(args.random_seed)
    embed_path = f'{os.path.splitext(args.input_sequence)[0]}.bert.npz'
    if not os.path.exists(embed_path):
        BertEmbedding(args.input_sequence, embed_path, args.pretrained_model, args.random_seed)
    if not os.path.exists(args.output_dir) and args.output_dir != "":
        os.makedirs(args.output_dir)
    for i, weights in enumerate(os.listdir(args.weights_dir)):
        model = joblib.load(os.path.join(args.weights_dir, weights))
        svm_predict(model, args.input_sequence, embed_path, os.path.join(args.output_dir, f'pred_result_{i+1}.txt'))
    if args.vote_required:
        vote_predict(args.input_sequence, args.output_dir, args.vote_threshold or 0.5)


def handle_pssm_cnn(args):
    set_seed(args.random_seed)
    embed_path = f'{os.path.splitext(args.input_sequence)[0]}.pssm.npz'
    PSSMEmbedding(args.pssm_directory, embed_path)
    if not os.path.exists(args.output_dir) and args.output_dir != "":
        os.makedirs(args.output_dir)
    model = PSSMCNN_tiny()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for i, weights in enumerate(os.listdir(args.weights_dir)):
        model.load_state_dict(torch.load(os.path.join(args.weights_dir, weights), map_location=device))
        dataset = TestDataset(args.input_sequence, embed_path, feature="MaaPSSM")
        predict(model, dataset, os.path.join(args.output_dir, f'pred_result_{i+1}.txt'), device)
    if args.vote_required:
        vote_predict(args.input_sequence, args.output_dir, args.vote_threshold or 0.5)


def handle_hybrid_bilstm(args):
    set_seed(args.random_seed)
    embed_path = f'{os.path.splitext(args.input_sequence)[0]}.bert.npz'
    if not os.path.exists(embed_path):
        BertEmbedding(args.input_sequence, embed_path, args.pretrained_model, args.random_seed)
    pssm_path = f'{os.path.splitext(args.input_sequence)[0]}.pssm30.npz'
    PSSMEmbedding(args.pssm_directory, pssm_path, pssm_length=30)
    if not os.path.exists(args.output_dir) and args.output_dir != "":
        os.makedirs(args.output_dir)
    model = BiLSTM_Attention(embedding_dim=100, hidden_dim=256, n_layers=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for i, weights in enumerate(os.listdir(args.weights_dir)):
        model.load_state_dict(torch.load(os.path.join(args.weights_dir, weights), map_location=device))
        dataset = HybridTestDataset(args.input_sequence, embed_path, pssm_path, 30)
        predict(model, dataset, os.path.join(args.output_dir, f'pred_result_{i+1}.txt'), device)
    if args.vote_required:
        vote_predict(args.input_sequence, args.output_dir, args.vote_threshold or 0.5)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='model')

    parser_1 = subparsers.add_parser(name='tapebert_mlp', help='TAPEBert_MLP')
    parser_1.add_argument('-pretrain', '--pretrained_model', default="weights/tape/bert-base")
    parser_1.set_defaults(func=handle_tapebert_mlp)

    parser_2 = subparsers.add_parser(name='tapebert_svm', help='TAPEBert_SVM')
    parser_2.add_argument('-pretrain', '--pretrained_model', default="weights/tape/bert-base")
    parser_2.set_defaults(func=handle_tapebert_svm)

    parser_3 = subparsers.add_parser(name='pssm_cnn', help='PSSM_CNN')
    parser_3.add_argument('-pssm', '--pssm_directory', default="pssm_files")
    parser_3.set_defaults(func=handle_pssm_cnn)

    parser_4 = subparsers.add_parser(name='hybrid_bilstm', help='HybridBiLSTM')
    parser_4.add_argument('-pretrain', '--pretrained_model', default="weights/tape/bert-base")
    parser_4.add_argument('-pssm', '--pssm_directory', default="pssm_files")
    parser_4.set_defaults(func=handle_hybrid_bilstm)

    parser.add_argument('-in', '--input_sequence', required=True)
    parser.add_argument('-weights', '--weights_dir', required=True)
    parser.add_argument('-seed', '--random_seed', default=42)
    parser.add_argument('-out', '--output_dir', default='results')
    parser.add_argument('-vote', '--vote_required', action='store_true')
    parser.add_argument('-threshold', '--vote_threshold', default=None, type=float)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
