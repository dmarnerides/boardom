from torch.utils.data import Dataset


class ListDataset(Dataset):
    def __init__(self, data, preprocess=None):
        self.data = data
        self.size = len(data)
        self.preprocess = preprocess

    def __getitem__(self, idx):
        datum = self.data[idx]
        if self.preprocess is not None:
            datum = self.preprocess(datum)
        return datum

    def __len__(self):
        return self.size
