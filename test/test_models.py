import torch
from torch import nn
import boardom as bd


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear = nn.Linear(3, 5)
        self.conv = nn.Conv2d(7, 11, (13, 15))
        self.just_param = nn.Parameter(torch.ones((17,)))
        self.linear_no_bias = nn.Linear(19, 21)
        self.linear_no_bias.bias = None
        self.conv_no_grad = nn.Conv2d(23, 27, 29)
        self.conv_no_grad.requires_grad_(False)


class TestCountParameters:
    def test_counts_params_for_nn(self):
        num_params_all = (
            3 * 5 + 5 + 7 * 11 * 13 * 15 + 11 + 17 + 19 * 21 + 23 * 27 * 29 * 29 + 27
        )
        assert bd.count_parameters(NN()) == num_params_all
