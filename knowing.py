import torch
import torch.nn as nn
from bert.bert import BERT
from transformers import BertTokenizer
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class AggregationModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AggregationModule, self).__init__()
        # Gate mechanism parameters
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, 1)
        # Sigmoid activation for gating
        self.sigmoid = nn.Sigmoid()

    def forward(self, Ps, Pk, relevance_scores):
        """
        Args:
            Ps: Sentence-pair prediction (batch_size, 1)
            Pk: Knowledge prediction (batch_size, 1)
            relevance_scores: BM25 scores (batch_size, m)
        Returns:
            final_prediction: Weighted prediction (batch_size, 1)
        """
        # Calculate gate value g
        hidden = F.relu(self.W1(relevance_scores))  # Shape: (batch_size, hidden_dim)
        g = self.sigmoid(self.W2(hidden))          # Shape: (batch_size, 1)

        # Final prediction
        final_prediction = Ps * (1 - g) + Pk * g
        return final_prediction

class KnowledgeEnhancedBERT(nn.Module):
    def __init__(self, sentence_bert, knowledge_bert, 
                 bert_tokenizer, embed_size, num_knowledge, hidden_dim = 200, dropout=0.1):
        super(KnowledgeEnhancedBERT, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load BERT for sentence pairs
        self.tokenizer = bert_tokenizer
        self.sentence_bert = sentence_bert
        self.fcPs = nn.Linear(embed_size, 1)
        
        # Load BERT for knowledge snippets
        self.knowledge_bert = knowledge_bert

        # Attention for knowledge aggregation
        self.attention_matrix = nn.Parameter(torch.randn(embed_size, embed_size))
        self.fcPk = nn.Linear(2 * embed_size, 1)
        
        # Initialize BM25 for knowledge selection
        self.num_knowledge = num_knowledge

        self.aggregation = AggregationModule(num_knowledge, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, knowledge_mask=None):
        # Tokenize sentence pair
        e_s = self.sentence_bert(
            inputs['input_ids'], 
            inputs['token_type_ids'], 
            inputs['attention_mask']
        )[:, 0, :]  # CLS token
        P_s = self.sigmoid(self.fcPs(e_s))

        # Knowledge retrieval using BM25
        relevance_input_ids = inputs['relevance_input_ids']
        relevance_token_type_ids = inputs['relevance_token_type_ids']
        relevance_attention_mask = inputs['relevance_attention_mask']
        relevance_scores = inputs['relevance_scores']
        # Encode retrieved knowledge snippets
        e_k = [[] for _ in range(relevance_scores.shape[0])]
        for i in range(relevance_scores.shape[0]):
            for j in range(self.num_knowledge):
                e_k_i = self.knowledge_bert(
                    relevance_input_ids[i, j], 
                    relevance_token_type_ids[i, j], 
                    relevance_attention_mask[i, j]
                )[:, 0, :]
                e_k[i].append(e_k_i)
            e_k[i] = torch.stack(e_k[i], dim=0).squeeze(1)
        e_k = torch.stack(e_k, dim=0).squeeze(1)
        
        # Apply attention to aggregate knowledge
        e_s_T = e_s.unsqueeze(1)  # (1, embed_size)
        alpha = torch.matmul(e_s_T, self.attention_matrix)  # (1, embed_size)
        alpha = torch.matmul(alpha, torch.transpose(e_k, 1, 2))  # (1, m)
        alpha = F.softmax(alpha, dim=-1)  # (1, m)
        e_k = torch.matmul(alpha, e_k)

        # Combine sentence representation with knowledge representation
        combined = torch.cat((e_s, e_k.squeeze(1)), dim=-1)
        P_k = self.sigmoid(self.fcPk(combined))

        # Get top n results
         

        # Final prediction
        return self.aggregation(P_s, P_k, relevance_scores)