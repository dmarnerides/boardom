import numpy as np
import cv2
import boardom as bd

# These functions assume inputs are hdr images in range [0, 1] with linear luminance
# Outputs are scaled to [0,1]


def culling(x, low=10, high=90, gamma=2.2):
    low, high = np.percentile(x, (low, high))
    return bd.map_range(np.clip(x, low, high)) ** (1 / gamma)


def exposure(x, exposure=0, gamma=2.2):
    x = np.clip(x * (2 ** exposure), 0, 1)
    return x ** (1 / gamma)


def reinhard(x, intensity=0.0, light_adapt=0.9, color_adapt=0.1, gamma=2.2):
    return cv2.createTonemapReinhard(
        gamma=gamma,
        intensity=intensity,
        light_adapt=light_adapt,
        color_adapt=color_adapt,
    ).process(x)


def drago(x, saturation=1.0, gamma=2.2, bias=0.85):
    tmo = cv2.createTonemapDrago(gamma=gamma, saturation=saturation, bias=bias)
    return tmo.process(x)


def mantiuk(x, saturation=1.0, scale=0.75, gamma=2.2):
    tmo = cv2.createTonemapMantiuk(gamma=gamma, saturation=saturation, scale=scale)
    return tmo.process(x)
