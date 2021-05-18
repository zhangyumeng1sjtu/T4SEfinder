import numpy as np
import torch
from torch.utils.data import Dataset
from Bio import SeqIO


class TestDataset(Dataset):

    def __init__(self, fasta, embed_path, feature):
        self.data = np.load(embed_path, allow_pickle=True)
        self.names = [record.name for record in SeqIO.parse(fasta, "fasta")]
        self.feature = feature
        
    def __getitem__(self, idx):
        name = self.names[idx]
        embed = torch.tensor(self.data[name].item()[self.feature]).float()
        return embed, name

    def __len__(self):
        return len(self.names)


class HybridTestDataset(Dataset):

    def __init__(self, fasta, embed_path, pssm_path, pssm_length): 
        self.names = [record.id for record in SeqIO.parse(fasta, "fasta")]
        self.embeddings = np.load(embed_path, allow_pickle=True)
        self.pssms = np.load(pssm_path, allow_pickle=True) 
        self.pssm_length = pssm_length

    def __getitem__(self, idx):
        name = self.names[idx]
        embed = torch.tensor(self.embeddings[name].item()["avg"]).float()
        pssm = torch.tensor(self.pssms[name].item()["PSSM"+str(self.pssm_length)]).float()
        return embed, pssm, name

    def __len__(self):
        return len(self.names)
        