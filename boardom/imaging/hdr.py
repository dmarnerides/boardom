import torch
import numpy as np
import scipy.interpolate
import boardom as bd

KINDS = ['linear', 'cubic', 'torchlinear']
TTYPES = [torch.float32, torch.float64, torch.float16]

PU_L = 31.9270
PU_H = 149.9244


class _PU_INTERPOLATOR:
    # We cache the interpolation function in this dictionary
    f = {}

    @staticmethod
    def evaluate(t_in, kind):
        if kind not in KINDS:
            raise RuntimeError(
                f'Unknown kind for pu_encode: {kind}' f'\nChoices: {KINDS}'
            )
        f = _PU_INTERPOLATOR.f

        is_array = bd.is_array(t_in)
        if kind == 'torchlinear':
            if is_array:
                t_in = torch.from_numpy(t_in)
            t_in = t_in.clamp(1e-5, 1e10).log10()

            # Account for device and dtype for mlp to automatically cast
            if t_in.dtype not in TTYPES:
                raise RuntimeError(
                    f'Invalid dtype: {t_in.dtype}\nExpected one of {TTYPES}'
                )
            key = (kind, t_in.device, t_in.dtype)
            if not f.get(key, False):
                x, y = bd.assets.pu_space
                interpolator = bd.Interp1d(x, y)
                interpolator.to(key[1])
                interpolator.type(key[2])
                f[key] = interpolator
            interpolator = f[key]
        else:
            if not is_array:
                t_in = t_in.numpy()
            t_in = np.log10(t_in.clip(1e-5, 1e10))

            key = (kind, t_in.dtype)
            if not f.get(key, False):
                x, y = bd.assets.pu_space
                x, y = x.numpy().astype(key[1]), y.numpy().astype(key[1])
                f[key] = scipy.interpolate.interp1d(x, y, kind=kind)
            interpolator = f[key]

        result = interpolator(t_in)
        if is_array and (not bd.is_array(result)):
            result = result.numpy()
        if (not is_array) and bd.is_array(result):
            result = torch.from_numpy(result)
        return result


# kind='linear' reproduces the original matlab version
# kind='cubic' uses cubic splines and it deviates slightly from matlab version
# kind='torchlinear' uses an torch functions for linear interpolation
def pu_encode(x, kind='linear'):
    return 255 * (_PU_INTERPOLATOR.evaluate(x, kind) - PU_L) / (PU_H - PU_L)
