from torch.utils.data import Dataset, DataLoader


def _collate(x):
    return x


def loader_process_pool(indexed_fn, length, num_workers=0, worker_init_fn=None):
    class Dset(Dataset):
        def __init__(self):
            super().__init__()

        def __getitem__(self, index):
            return indexed_fn(index)

        def __len__(self):
            return length

    return DataLoader(
        Dset(),
        batch_size=1,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        collate_fn=_collate,
    )
