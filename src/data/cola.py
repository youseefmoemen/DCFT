from torch.utils.data import Dataset
from datasets import load_dataset


class Cola(Dataset):
    def __init__(self, split):
        super().__init__()
        self.split = split
        self.data = load_dataset('nyu-mll/glue', 'cola', split=self.split)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return super().__getitem__(index)
        
