import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_size, max_position_embeddings, type_vocab_size):
        super(BERTEmbeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embed_size)
        self.segment_embeddings = nn.Embedding(type_vocab_size, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, segment_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        segment_embeds = self.segment_embeddings(segment_ids)
        embeddings = token_embeds + position_embeds + segment_embeds
        return self.dropout(self.layer_norm(embeddings))

class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by the number of heads"
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query, mask):
        N, seq_len, embed_size = query.size()
        queries = self.query(query).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(key).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(value).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, values)
        out = out.transpose(1, 2).contiguous().view(N, seq_len, embed_size)
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class BERT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_position_embeddings, forward_expansion, dropout, type_vocab_size):
        super(BERT, self).__init__()
        self.embeddings = BERTEmbeddings(vocab_size, embed_size, max_position_embeddings, type_vocab_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, segment_ids, mask):
        x = self.embeddings(input_ids, segment_ids)
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x

class MaskedLanguageModel(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super(MaskedLanguageModel, self).__init__()
        self.linear = nn.Linear(embed_size, embed_size)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.norm(x)
        return self.fc(x)

class PretrainingBERT(nn.Module):
    def __init__(self, bert, vocab_size):
        super(PretrainingBERT, self).__init__()
        self.bert = bert
        self.mlm_head = MaskedLanguageModel(bert.embeddings.token_embeddings.embedding_dim, vocab_size)

    def forward(self, inputs):
        input_ids, segment_ids, mask = inputs
        x = self.bert(input_ids, segment_ids, mask)
        mlm_logits = self.mlm_head(x)
        return mlm_logits