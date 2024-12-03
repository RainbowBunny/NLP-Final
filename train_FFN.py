from FFN.FFN import FFNModel
from FFN.FFNDataset import FFNDataset
import torch
from torch.utils.data import DataLoader
from utils.trainer import Trainer, CheckPointArgs, TrainArgs

BATCH_SIZE = 16
model_name = "FFN"
experiment_name = "quora_dataset"

nb_words = 112955

data = torch.load("data.pt")
q1 = data['q1']
q2 = data['q2']
hcf = data['hcf']
values = data['values'].double()

print(q1.shape)

word_embedding_matrix = torch.load("word_embedding_matrix.pt")['word_embedding_matrix']

train_indices = range(0, 320000)
valid_indices = range(320000, 400000)

train_dataset = FFNDataset(q1, q2, hcf, values, train_indices)
valid_dataset = FFNDataset(q1, q2, hcf, values, valid_indices)

training_args = TrainArgs(num_epochs = 100, batch_size = BATCH_SIZE, learning_rate = 1e-4)
checkpoint_args = CheckPointArgs(model_name, experiment_name)

model = FFNModel(nb_words + 1, 300, word_embedding_matrix, 12).double()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return {
        "Total": total_params,
        "Trainable": trainable_params,
        "Non-trainable": non_trainable_params
    }

# Get parameter count
param_counts = count_parameters(model)
print(f"Total Parameters: {param_counts['Total']}")
print(f"Trainable Parameters: {param_counts['Trainable']}")
print(f"Non-trainable Parameters: {param_counts['Non-trainable']}")
# trainer = Trainer(model, train_dataset, valid_dataset, checkpoint_args, training_args)

# trainer.train()