import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

checkpoint_FFN = torch.load("checkpoints/FFN_quora_dataset_checkpoint.pth")
checkpoint_knowing = torch.load("checkpoints/knowing_quora_dataset_checkpoint.pth")
checkpoint_pretrain = torch.load("checkpoints/pretrain_bert_bookcropus_checkpoint.pth")

epochs = range(1, 31)

# Plotting
plt.figure(figsize=(12, 8))

# Subplot for accuracy
plt.subplot(1, 3, 1)
plt.plot(epochs, checkpoint_FFN['train_cel'][:30], label="Train FFN CEL", marker='o')
plt.plot(epochs, checkpoint_FFN['valid_cel'][:30], label="Eval FFN CEL", marker='s')
plt.title("FFN Cel")
plt.xlabel("Epoch")
plt.ylabel("CEL")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs, checkpoint_knowing['train_cel'][:30], label="Train Knowing CEL", marker='o')
plt.plot(epochs, checkpoint_knowing['valid_cel'][:30], label="Eval Knowing CEL", marker='s')
plt.title("Knowing Cel")
plt.xlabel("Epoch")
plt.ylabel("CEL")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epochs, checkpoint_pretrain['train_cel'][:30], label="Train PretrainBERT CEL", marker='o')
plt.plot(epochs, checkpoint_pretrain['valid_cel'][:30], label="Eval PretrainBERT CEL", marker='s')
plt.title("PretrainBERT Cel")
plt.xlabel("Epoch")
plt.ylabel("CEL")
plt.legend()

plt.tight_layout()
plt.show()