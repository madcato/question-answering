import torch

class GroceryGourmetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, decodings):
        self.encodings = encodings
        self.decodings = decodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.decodings[idx])
        return item

    def __len__(self):
        return len(self.decodings)
