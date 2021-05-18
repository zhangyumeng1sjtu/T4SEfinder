import os
from collections import defaultdict

from Bio import SeqIO
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader

from .utils import read_pssm


def BertEmbedding(fasta, embed_path, pretrained_model, random_seed):

    embed_dir = os.path.split(embed_path)[0]
    if not os.path.exists(embed_dir) and embed_dir != "":
        os.makedirs(embed_dir)

    os.system(f'tape-embed transformer {fasta} {embed_path} {pretrained_model} --tokenizer iupac \
                --batch_size 1 --seed {random_seed} --log_level WARNING')


def PSSMEmbedding(pssm_dir, embed_path, pssm_length=None):

    embed_dir = os.path.split(embed_path)[0]
    if not os.path.exists(embed_dir) and embed_dir != "":
        os.makedirs(embed_dir)

    output = defaultdict(dict)
    for filename in os.listdir(pssm_dir):
        if filename.endswith(".pssm"):
            result = read_pssm(os.path.join(pssm_dir, filename), length=pssm_length)
            keyname = os.path.splitext(filename)[0]
            output[keyname] = result

    np.savez(embed_path, **output)


def predict(classifier, dataset, outfile, device):

    dataloader = DataLoader(dataset, shuffle=False, batch_size=max(len(dataset), 100))

    model = classifier.to(device)
    model.eval()
    with torch.no_grad():
        result = {}
        for data in dataloader:

            if len(data) == 3:
                vector, pssm, names = data
                vector, pssm = vector.to(device), pssm.to(device)
                output, _ = model(vector, pssm)
            else:
                input, names = data
                input = input.to(device)
                output = model(input)
            
            probs = torch.softmax(output, dim=1)[:, 1]
            probs = probs.cpu().numpy()
            for name, prob in zip(names, probs):
                result[name] = prob

        with open(outfile, "w") as f:
            for name, prob in result.items():
                label = "+1" if prob >= 0.5 else "-1"
                f.write(f'{name}\t{prob:.3f}\t{label}\n')


def svm_predict(classifier, fasta, embed_path, outfile):

    embeddings = np.load(embed_path, allow_pickle=True)
    embed_vector = np.array([embeddings[record.id].item()['avg'] for record in SeqIO.parse(fasta, "fasta")])
    probs = classifier.predict_proba(embed_vector)[:, 1]
    names = np.array([record.id for record in SeqIO.parse(fasta, "fasta")])
    result = {}
    for name, prob in zip(names, probs):
        result[name] = prob

    with open(outfile, "w") as f:
        for name, prob in result.items():
            label = "+1" if prob >= 0.5 else "-1"
            f.write(f'{name}\t{prob:.3f}\t{label}\n')


def vote_predict(fasta, result_dir, threshold=0.5):

    vote_result = defaultdict(list)

    fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta, "fasta"))
    T4SEs = []

    for i in range(5):
        pred_result = f"pred_result_{i+1}.txt"
        for line in open(os.path.join(result_dir, pred_result), 'r'):
            name, prob, label = line.strip().split()
            vote_result[name].append(label)

    with open(os.path.join(result_dir, "pred_result_vote.txt"), "w") as f:
        for key, value in vote_result.items():
            prob = value.count("+1")/len(value)
            label = "+1" if prob >= threshold else "-1"
            if label == "+1":
                T4SEs.append(fasta_dict[key])
            f.write("%s\t%.3f\t%s\n" % (key, prob, label))

    SeqIO.write(T4SEs, os.path.join(result_dir, "T4SEs.fasta"), "fasta")
