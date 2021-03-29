import os
import torch
import numpy as np
from collections.abc import Mapping, Sequence
import logging
import boardom as bd


def _check_valid_fields(fields):
    for field in fields:
        if not isinstance(field, str):
            msg = f'CSVLogger received invalid field type: {type(field)}\nExpected a string.'
            raise RuntimeError(msg)
        if not field.isidentifier():
            msg = f'CSVLogger received invalid field: {field}\nMust be a valid Python identifier.'
            raise RuntimeError(msg)


def _get_item(value):
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return value.item()
    else:
        return value


class CSVLogger:
    def __init__(self, name, fields=None, directory=".", delimiter=',', resume=True):
        if fields is None:
            fields = ['value']
        _check_valid_fields(fields)
        if not name.endswith('.csv'):
            name = name + '.csv'
        self.directory = bd.process_path(directory, True)
        self.file = os.path.join(self.directory, name)
        self.fields = fields
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        # Write header
        if not resume or not os.path.exists(self.file):
            with open(self.file, 'w') as f:
                f.write(delimiter.join(fields) + '\n')

        file_handler = logging.FileHandler(self.file)
        # Adding underscore to avoid clashes with reserved words from logging
        field_tmpl = delimiter.join([f'{{_{x}}}' for x in fields])

        file_handler.setFormatter(logging.Formatter(field_tmpl, style='{'))
        self.logger.addHandler(file_handler)

    def __call__(self, values):
        """Same as :meth:`log`"""
        self.log(values)

    def _create_dict(self, values):
        if isinstance(values, Sequence):
            return {f'_{key}': val for key, val in zip(self.fields, values)}
        elif isinstance(values, Mapping):
            return {
                f'_{key}': _get_item(values[key]) if key in values else None
                for key in self.fields
            }
        else:
            return {f'_{self.fields[0]}': values}

    def log(self, values):
        """Logs a row of values.

        Args:
            values (dict): Dictionary containing the names and values.
        """
        if not values:
            return
        self.logger.info('', extra=self._create_dict(values))
