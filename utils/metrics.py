import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    cel = torch.nn.CrossEntropyLoss()

    a = torch.randn((50))
    b = torch.randn((50))

    print(cel(a, b))