from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from transformers import BertTokenizer

class QuoraDataset(Dataset):
    def __init__(self, questions1, questions2, labels, tokenizer, max_len=128):
        self.questions1 = questions1
        self.questions2 = questions2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        q1 = str(self.questions1[index])
        q2 = str(self.questions2[index])
        label = self.labels[index]

        # Tokenize the questions
        inputs = self.tokenizer(
            q1, q2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

if __name__ == '__main__':
    data = pd.read_csv("train_small.csv")  # Replace with your dataset file path

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Features: `question1` and `question2`
    questions1 = data['question1'].values
    questions2 = data['question2'].values

    # Target: `is_duplicate`
    labels = data['is_duplicate'].values

    print(questions1[0], questions2[0], labels[0])

    dataset = QuoraDataset(questions1, questions2, labels, )