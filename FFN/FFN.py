import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_weights, hcf_size):
        super(FFNModel, self).__init__()
        
        # Shared Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_weights, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Freeze embeddings
        
        # TimeDistributed Dense Layer Equivalent
        self.dense = nn.Linear(embedding_dim, embedding_dim)
        
        # HCF Branch
        self.hcf_reshape = nn.Flatten()  # Reshape HCF features
        
        # Fully Connected Layers after concatenation
        self.fc1 = nn.Linear(embedding_dim * 2 + hcf_size, 200, dtype=torch.float32)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        self.output = nn.Linear(200, 1)
        
        # Normalization and Dropout Layers
        self.batchnorm1 = nn.BatchNorm1d(200)
        self.batchnorm2 = nn.BatchNorm1d(200)
        self.batchnorm3 = nn.BatchNorm1d(200)
        self.batchnorm4 = nn.BatchNorm1d(200)
        self.dropout = nn.Dropout(0.2)
    
    def time_distributed_dense(self, x):
        # Apply Dense Layer over each time step
        return F.relu(self.dense(x))
    
    def forward(self, inputs):
        q1, q2, hcf = inputs["q1"], inputs["q2"], inputs["hcf"]
        # Q1 Branch
        q1_emb = self.embedding(q1)  # (batch_size, max_seq_len, embedding_dim)
        q1_dense = self.time_distributed_dense(q1_emb)  # Apply Dense layer
        q1_sum = torch.sum(q1_dense, dim=1)  # Sum over the time dimension
        
        # Q2 Branch
        q2_emb = self.embedding(q2)
        q2_dense = self.time_distributed_dense(q2_emb)
        q2_sum = torch.sum(q2_dense, dim=1)
        
        # HCF Branch
        hcf_flat = self.hcf_reshape(hcf)
        # Concatenate Q1, Q2, and HCF outputs
        merged = torch.cat([q1_sum, q2_sum, hcf_flat], dim=1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(merged))
        x = self.batchnorm1(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc4(x))
        x = self.batchnorm4(x)
        x = self.dropout(x)
        
        # Final Output Layer
        out = torch.sigmoid(self.output(x))
        return out
