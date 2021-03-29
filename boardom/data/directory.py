import os
from torch.utils.data import Dataset
import boardom as bd


class DirectoryDataset(Dataset):
    """Creates a dataset of images (no label) recursively (no structure requirement).
    
    Similar to `torchvision.datasets.FolderDataset`, however there is no need for
    a specific directory structure, or data format.

    Args:
        data_root_path (string): Path to root directory of data.
        extensions (list or tuple): Extensions/ending patterns of data files.
        loader (callable): Function that loads the data files.
        preprocess (callable, optional): A function that takes a single data
            point from the dataset to preprocess on the fly (default None).

    """

    def __init__(self, data_root_path, data_extensions, load_fn, preprocess=None):
        super().__init__()
        data_root_path = bd.process_path(data_root_path)
        self.file_list = []
        for root, _, fnames in sorted(os.walk(data_root_path)):
            for fname in fnames:
                if any(
                    fname.lower().endswith(extension) for extension in data_extensions
                ):
                    self.file_list.append(os.path.join(root, fname))
        if len(self.file_list) == 0:
            msg = 'Could not find any files with extensions:\n[{0}]\nin\n{1}'
            raise RuntimeError(msg.format(', '.join(data_extensions), data_root_path))

        self.preprocess = preprocess
        self.load_fn = load_fn

    def __getitem__(self, index):
        dpoint = self.load_fn(self.file_list[index])
        if self.preprocess is not None:
            dpoint = self.preprocess(dpoint)
        return dpoint

    def __len__(self):
        return len(self.file_list)
