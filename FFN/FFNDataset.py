from torch.utils.data import Dataset

class FFNDataset(Dataset):
    def __init__(self, question_1_id, question_2_id, hcf, values, indices):
        self.question_1_id = question_1_id
        self.question_2_id = question_2_id
        self.hcf = hcf
        self.values = values
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]

        return {
            'q1': self.question_1_id[index],
            'q2': self.question_2_id[index],
            'hcf': self.hcf[index]
        }, self.values[index]