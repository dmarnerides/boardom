from copy import deepcopy
from functools import partial
from tqdm import tqdm
from torch.utils.data import DataLoader
import boardom as bd


def get_decoder_fn(to_decode):
    def ret(x):
        if isinstance(x, (tuple, list)):
            return type(x)(
                [bd.decode_loaded(y) if i in to_decode else y for i, y in enumerate(x)]
            )
        else:
            return bd.decode_loaded(x)

    return ret


# 1. If we don't encode-decode then on_load is called and preprocess is on the fly
# 2. If we encode, the orig_dataset must provide encoded data. on_load is overwritten
#    and decode + preprocess happens on the fly


# If to_decode is not None, it must be a list, containing all the occurances
# of encoded images

# If compress_lvl is in 1-9 the objects are compressed in memory (in self._loaded_set)
# and decompressed on the fly
class LoadedDataset:
    def __init__(
        self,
        orig_dataset,
        preprocess=None,
        on_load=None,
        decode=False,
        decode_positions=[0],
        num_workers=0,
        compress_lvl=0,
    ):
        super().__init__()
        assert compress_lvl <= 9 and compress_lvl >= 0
        self.orig_dataset = orig_dataset

        preprocess = bd.identity if preprocess is None else preprocess
        if decode:
            on_load = bd.identity
            preprocess = bd.compose(get_decoder_fn(decode_positions), preprocess)
        else:
            on_load = bd.identity if on_load is None else on_load

        if compress_lvl > 0:
            on_load = bd.compose(on_load, partial(bd.compress, level=compress_lvl))
            preprocess = bd.compose(bd.decompress, preprocess)

        # Use dataloader to load the dataset
        dummy_loader = DataLoader(
            orig_dataset,
            batch_size=1,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=bd.identity,
        )
        # DataLoader returns a list, so take 0th element
        # Copy to avoid shared memory issues
        loaded_dataset = [
            on_load(deepcopy(x[0])) for x in tqdm(dummy_loader, desc='Loading: ')
        ]
        self.loaded_dataset = loaded_dataset
        self.preprocess = preprocess

    def __getitem__(self, index):
        return self.preprocess(self.loaded_dataset[index])

    def __len__(self):
        return len(self.loaded_dataset)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.orig_dataset, attr)
