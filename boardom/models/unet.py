from .module import Module, magic_module


class OriginalUNet(Module):
    def __init__(self, in_features=1, out_features=2, nf=[64, 128, 256, 512, 1024]):
        super().__init__()
        bottleneck = [
            ['maxpool2d', 2, 2],
            ['conv2d', nf[3], nf[4], 3],
            ['relu', True],
            ['conv2d', nf[4], nf[4], 3],
            ['relu', True],
            ['convtranspose2d', nf[4], nf[3], 2, 2],
        ]
        level_4 = self._build_level(nf[2], nf[3], bottleneck)
        level_3 = self._build_level(nf[1], nf[2], level_4)
        level_2 = self._build_level(nf[0], nf[1], level_3)
        level_1 = self._build_level(in_features, nf[0], level_2)
        # Remove first maxpool and last conv2d
        ret = level_1[1:-1] + [['conv2d', nf[0], out_features, 1]]
        self.main = magic_module(ret)

    def _build_level(self, n_in, n_out, sublevel):
        return [
            ['maxpool2d', 2, 2],
            ['conv2d', n_in, n_out, 3],
            ['relu', True],
            ['conv2d', n_out, n_out, 3],
            ['relu', True],
            ['split', ['identity'], sublevel],
            ['cropcat2d'],
            ['conv2d', 2 * n_out, n_out, 3],
            ['relu', True],
            ['conv2d', n_out, n_out, 3],
            ['relu', True],
            ['convtranspose2d', n_out, n_in, 2, 2],
        ]

    def forward(self, x):
        return self.main(x)
