def grow_dataset(orig_dataset, growth_factor):
    orig_len = len(orig_dataset)
    new_len = int(growth_factor * orig_len)
    orig_class = orig_dataset.__class__

    class GrownDataset(orig_class):
        def __len__(self):
            return new_len

        def __getitem__(self, index):
            return orig_class.__getitem__(self, index % orig_len)
            #  return super().__getitem__(index % orig_len)

        def ungrow(self):
            self.__class__ = orig_class

    orig_dataset.__class__ = GrownDataset

    return orig_dataset
