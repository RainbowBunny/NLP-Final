import torch
from bert.bertDataset import MLMBERTDataset
from torch.utils.data import DataLoader
from bert.bert import PretrainingBERT
from bert.bert import BERT
from transformers import BertTokenizer
from utils.pretrainer import Trainer, CheckPointArgs, TrainArgs

vocab_size = 30522
embed_size = 256
num_heads = 4
num_layers = 4
max_position_embeddings = 256
forward_expansion = 4
dropout = 0.1
type_vocab_size = 2
BATCH_SIZE = 16
model_name = "pretrain_bert"
experiment_name = "bookcropus"

bert_model = BERT(vocab_size, embed_size, num_heads, num_layers, max_position_embeddings, forward_expansion, dropout, type_vocab_size)
pretraining_model = PretrainingBERT(bert_model, vocab_size)

ds = torch.load("masklm_input.pt")
inputs = ds["inputs"]
token_id = ds["token_id"]
attention_mask = ds["attention_mask"]
targets = ds["targets"]

train_indices = range(0, 32000)
valid_indices = range(32000, 40000)


train_dataset = MLMBERTDataset(inputs, token_id, attention_mask, targets, train_indices)
valid_dataset = MLMBERTDataset(inputs, token_id, attention_mask, targets, valid_indices)

training_args = TrainArgs(num_epochs = 100, batch_size = BATCH_SIZE, learning_rate = 1e-4)
checkpoint_args = CheckPointArgs(model_name, experiment_name)

trainer = Trainer(pretraining_model, train_dataset, valid_dataset, checkpoint_args, training_args)

trainer.train()