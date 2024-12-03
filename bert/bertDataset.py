from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch

class MLMBERTDataset(Dataset):
    def __init__(self, masked_id, token_id, attention_mask, trues_id, indices):
        self.masked_id = masked_id
        self.token_id = token_id
        self.attention_mask = attention_mask
        self.trues_id = trues_id
        self.indices = indices
        

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]

        return [
            self.masked_id[index], self.token_id[index], self.attention_mask[index]
        ], self.trues_id[index]

class QuoraDataset(Dataset):
    def __init__(self, question_1, question_2, values, bm25_index, bm25_score, indices):
        self.question_1 = question_1
        self.question_2 = question_2
        self.values = values
        self.indices = indices
        self.bm25_index = bm25_index
        self.bm25_score = bm25_score
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        question_pair = str(self.question_1[index]) + " [SEP] " + str(self.question_2[index])
        token = self.tokenizer(question_pair, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
        bm25_input_ids = []
        bm25_token_type_ids = []
        bm25_attention_mask = []
        for j in range(len(self.bm25_index[index])):
            potato = self.tokenizer(self.bm25_index[index, j]['text'], max_length=128, truncation=True, padding="max_length", return_tensors="pt")
            bm25_input_ids.append(potato['input_ids'])
            bm25_token_type_ids.append(potato['token_type_ids'])
            bm25_attention_mask.append(potato['attention_mask'])
        bm25_input_ids = torch.stack(bm25_input_ids, dim=0)
        bm25_token_type_ids = torch.stack(bm25_token_type_ids, dim=0)
        bm25_attention_mask = torch.stack(bm25_attention_mask, dim=0)

        return {
            'input_ids': token['input_ids'].squeeze(0),
            'token_type_ids': token['token_type_ids'].squeeze(0),
            'attention_mask': token['attention_mask'].squeeze(0),
            'relevance_input_ids': bm25_input_ids,
            'relevance_token_type_ids': bm25_token_type_ids,
            'relevance_attention_mask': bm25_attention_mask,
            'relevance_scores': self.bm25_score[index]
        }, self.values[index]