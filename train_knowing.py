import torch
import pandas as pd
from bert.bertDataset import QuoraDataset
from bert.bert import PretrainingBERT, BERT
from knowing import KnowledgeEnhancedBERT
from transformers import BertTokenizer
from utils.trainer import Trainer, CheckPointArgs, TrainArgs
import pickle

vocab_size = 30522
embed_size = 256
num_heads = 4
num_layers = 4
max_position_embeddings = 256
forward_expansion = 4
dropout = 0.1
type_vocab_size = 2
BATCH_SIZE = 16
num_knowledge = 5
model_name = "knowing"
experiment_name = "quora_dataset"

bert_model = BERT(vocab_size, embed_size, num_heads, num_layers, max_position_embeddings, forward_expansion, dropout, type_vocab_size)
pretraining_model = PretrainingBERT(bert_model, vocab_size)
loaded_checkpoint = torch.load("checkpoints/pretrain_bert_bookcropus_checkpoint.pth", map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
pretraining_model.load_state_dict(loaded_checkpoint["best_model"])
sentence_bert = pretraining_model.bert

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

bert_model_2 = BERT(vocab_size, embed_size, num_heads, num_layers, max_position_embeddings, forward_expansion, dropout, type_vocab_size)
pretraining_model_2 = PretrainingBERT(bert_model, vocab_size)
loaded_checkpoint_2 = torch.load("checkpoints/pretrain_bert_bookcropus_checkpoint.pth", map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
pretraining_model_2.load_state_dict(loaded_checkpoint["best_model"])
knowledge_bert = pretraining_model_2.bert

model = KnowledgeEnhancedBERT(sentence_bert, knowledge_bert, tokenizer, embed_size, num_knowledge = 5)

train_indices = range(0, 32000)
valid_indices = range(32000, 40000)

df = pd.read_csv("train.csv")
question_1 = df['question1']
question_2 = df['question2']
values = torch.tensor(df['is_duplicate']).double()

with open('bm25_index.pickle', 'rb') as handle:
    bm25_index = pickle.load(handle)
with open('bm25_score.pickle', 'rb') as handle:
    bm25_score = pickle.load(handle)

train_dataset = QuoraDataset(question_1, question_2, values, bm25_index, bm25_score, train_indices)
valid_dataset = QuoraDataset(question_1, question_2, values, bm25_index, bm25_score, valid_indices)

training_args = TrainArgs(num_epochs = 100, batch_size = BATCH_SIZE, learning_rate = 1e-4)
checkpoint_args = CheckPointArgs(model_name, experiment_name)

trainer = Trainer(model, train_dataset, valid_dataset, checkpoint_args, training_args)

trainer.train()