import torch


def _get_magnitude(real, imag):
    return (real.pow(2) + imag.pow(2)).sqrt()


def _get_phase(real, imag, eps=1e-15):
    return (imag / (real + eps)).atan()


def imfft(img, normalized=False, onesided=False, mode='magnitude_phase', eps=1e-15):
    fft = torch.rfft(img, 2, normalized=normalized, onesided=onesided)
    if mode == 'fft':
        return fft
    real, imag = fft[..., 0], fft[..., 1]
    if mode == 'real_imag':
        return real, imag
    elif mode == 'magnitude_phase':
        return _get_magnitude(real, imag), _get_phase(real, imag, eps=1e-15)
    else:
        raise ValueError(f'Invalid fft mode {mode}')
