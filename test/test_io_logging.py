import torch
from torch import nn
import numpy as np
import boardom as bd
from importlib import reload
import pytest


class TestWrite:
    def test_empty_prints_newline(self, capsys):
        bd.write()
        captured = capsys.readouterr()
        assert captured.out == "\n"

    def test_can_print_objects(self, capsys):
        for obj in [torch.tensor([1]), np.array([1]), {'foo': 1}]:
            bd.write(obj)
            captured = capsys.readouterr()
            assert captured.out == str(obj) + "\n"

    def test_can_print_multiple_args(self, capsys):
        args = [1, 2, {'foo': 3}]
        bd.write(*args)
        captured = capsys.readouterr()
        assert captured.out == ' '.join([str(x) for x in args]) + "\n"

    def test_can_change_separator(self, capsys):
        args = [1, 2, {'foo': 3}]
        bd.write(*args, sep=', ')
        captured = capsys.readouterr()
        assert captured.out == ', '.join([str(x) for x in args]) + "\n"

    def test_can_change_end_str(self, capsys):
        args = [1, 2, {'foo': 3}]
        bd.write(*args, end='')
        captured = capsys.readouterr()
        assert captured.out == ' '.join([str(x) for x in args])
