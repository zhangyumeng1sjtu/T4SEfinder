import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):

    def __init__(self, 
                 in_dim: int = 768,
                 hid_dim: int = 120,
                 num_class: int = 2,
                 dropout: float = 0.5):
        super(SimpleMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(hid_dim, num_class)
        )

    def forward(self, x):
        return self.mlp(x)


class PSSMCNN_tiny(nn.Module):

    def __init__(self, num_class=2):
        super(PSSMCNN_tiny, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=(1,1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=(1,1))
        self.maxpool = nn.MaxPool2d((2,2), stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(5*5*64,120)
        self.fc2 = nn.Linear(120,num_class)
        self.dropout = nn.Dropout(0.3)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.maxpool(F.relu(self.bn2(self.conv2(x))))
        x = x.contiguous().view(x.size(0), -1)
        # x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PSSMEmbeddingLayer(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 method: str):
        super(PSSMEmbeddingLayer, self).__init__()
        self.layer = nn.ModuleDict({
            'conv': nn.Sequential(nn.Conv1d(1, embedding_dim//20, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm1d(embedding_dim//20)),
            'linear': nn.Sequential(nn.Linear(20, embedding_dim), nn.Dropout(0.1)),
            'upsample': nn.Sequential(nn.Upsample(scale_factor=embedding_dim//20, mode='nearest'), nn.Dropout(0.1))
        })
        self.method = method

    def forward(self, x):
        batch, length, _ = x.size()
        x = x.contiguous().view(batch*length, 1, -1) # [batch*length, 1, 20]
        if self.method == 'linear':
            x = self.layer[self.method](x.squeeze(1))
        else:
            x = self.layer[self.method](x)
        x = x.contiguous().view(batch, length, -1)
        return x


class BiLSTM_Attention(nn.Module):

    def __init__(self, 
                 embedding_dim:int,
                 hidden_dim: int,
                 n_layers: int,
                 embed_method: str = "conv"):
        super(BiLSTM_Attention, self).__init__()
        assert embedding_dim % 20 == 0
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.pssm_embed = PSSMEmbeddingLayer(embedding_dim, embed_method)
        self.extractor = nn.Linear(768, hidden_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim * 2, 2)
        self.dropout = nn.Dropout(0.1)

    # x, query: [batch, seq_len, hidden_dim*2]
    @staticmethod
    def attention_layer(x, query):      
        d_k = query.size(-1)                                              
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  # scores:[batch, seq_len, seq_len]
        p_attn = F.softmax(scores, dim = -1)                              
        context = torch.matmul(p_attn, x).sum(1)       # [batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn


    def forward(self, vector, pssm):

        feature = self.extractor(vector).unsqueeze(0)
        hidden = self.dropout(torch.cat([feature for i in range(2*self.n_layers)], 0))

        embedding = self.pssm_embed(pssm)

        output, (hidden, cell) = self.rnn(embedding, (hidden, hidden))

        query = self.dropout(output)
        attn_output, attention = self.attention_layer(output, query)       
        final_output = self.fc(attn_output)

        return final_output, attention
