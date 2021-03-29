import torch


def _get_mean_std(x, mean_list, std_list):
    mean = torch.tensor(mean_list, device=x.device, dtype=x.dtype)
    std = torch.tensor(std_list, device=x.device, dtype=x.dtype)
    if x.ndim == 3:
        mean, std = mean[:, None, None], std[:, None, None]
    elif x.ndim == 4:
        mean, std = mean[None, :, None, None], std[None, :, None, None]
    else:
        raise RuntimeError('Expected 3 or 4 dimensional tensor but got shape {x.shape}')
    return mean, std


def imnormalize(x, mean, std):
    mean, std = _get_mean_std(x, mean, std)
    return (x - mean) / std


def imdenormalize(x, mean, std):
    mean, std = _get_mean_std(x, mean, std)
    return x * std + mean


def normalize_torchvision_imagenet(x):
    return imnormalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def denormalize_torchvision_imagenet(x):
    return imdenormalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).clamp(0, 1)
