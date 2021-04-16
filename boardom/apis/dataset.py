from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import boardom as bd


def _custom_get_item(self, index):
    img, target = self.data[index], self.targets[index]
    if self.transform is not None:
        img = self.transform(img)
    if self.target_transform is not None:
        target = self.target_transform(target)
    return img, target


_dsets = {
    'mnist': 'MNIST',
    'fashionmnist': 'FashionMNIST',
    'cifar10': 'CIFAR10',
    'cifar100': 'CIFAR100',
}
_NEED_GETITEM = ['mnist', 'fashionmnist', 'cifar10', 'cifar100']
_NEED_UNSQUEEZING = ['mnist', 'fashionmnist']


@bd.autoconfig
def torchvision_dataset(
    torchvision_dataset,
    data_root_path,
    download=True,
    transform=None,
    target_transform=None,
    train=True,
    torchvision_as_is=True,
):
    """Creates a dataset from torchvision, configured using Command Line Arguments.

    Args:
        transform (callable, optional): A function that transforms an image (default None).
        target_transform (callable, optional): A function that transforms a label (default None).
        train (bool, optional): Training set or validation - if applicable (default True).
        subset (string, optional): Specifies the subset of the relevant
            categories, if any of them was split (default, None).

    Relevant Command Line Arguments:

        - **dataset**: `--data`, `--torchvision_dataset`.

    Warning:
        Unlike the torchvision datasets, this function returns a dataset that
        uses NumPy Arrays instead of a PIL Images.
    """
    if torchvision_dataset is None:
        raise RuntimeError('Argumnent torchvision_dataset was not specified.')
    dset_str = torchvision_dataset.lower()
    bd.log(f'Using {dset_str} dataset from torchvision.')
    if dset_str in _dsets:
        TVDataset = getattr(datasets, _dsets[dset_str])
        if dset_str in _NEED_GETITEM:
            TVDataset.__getitem__ = _custom_get_item
        ret_dataset = TVDataset(
            data_root_path,
            train=train,
            download=download,
            transform=transform,
            target_transform=target_transform,
        )
        if dset_str in _NEED_UNSQUEEZING:
            ret_dataset.data = ret_dataset.data.unsqueeze(3).numpy()
            ret_dataset.targets = ret_dataset.targets.numpy()
    else:
        raise NotImplementedError(f'{torchvision_dataset} dataset not implemented.')
    return ret_dataset


@bd.autoconfig(
    ignore=[
        'dataset',
        'preprocess',
        'worker_init_fn',
        'collate_fn',
        'sampler',
        'batch_sampler',
    ]
)
def loader(
    dataset,
    batch_size=1,
    shuffle=True,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    prefetch_factor=2,
    persistent_workers=False,
    preprocess=None,
):
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'sampler': sampler,
        'batch_sampler': batch_sampler,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
        'timeout': timeout,
        'worker_init_fn': worker_init_fn,
        'prefetch_factor': prefetch_factor,
        'persistent_workers': persistent_workers,
    }
    collate_fn = collate_fn or default_collate
    if preprocess is not None:

        def get_collate(_old_collate):
            def collate_new(*args, **kwargs):
                return preprocess(_old_collate(*args, **kwargs))

            return collate_new

        collate_fn = get_collate(collate_fn)
    kwargs['collate_fn'] = collate_fn
    return DataLoader(dataset, **kwargs)
