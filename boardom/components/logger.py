import os
from collections.abc import Mapping, Sequence, Callable
from torch.utils.tensorboard import SummaryWriter
import boardom as bd
from .util import _create_state_dict_element, _prepare_cfg


class AverageTracker:
    def __init__(self, engine, size_key, fields):
        self.trackers = {}
        self.engine = engine
        self.size_key = size_key
        self.fields = fields

    def update(self, values):
        size = 1
        if self.size_key is not None:
            size = self.engine[self.size_key]
        fields = values.keys() if self.fields is None else self.fields
        trackers = self.trackers
        for f in fields:
            if f not in values:
                continue
            tracker = trackers.get(f, bd.Average())
            tracker.add(values[f], size)
            trackers[f] = tracker

    def reset(self):
        for tracker in self.trackers.values():
            tracker.reset()

    def get(self):
        ret = {k: t.get() for k, t in self.trackers.items()}
        return {k: v for k, v in ret.items() if v is not None}


class GenericLogger:
    def __init__(self, get_value_fn, log_fn, state_key, fields):
        self._get_value_fn = get_value_fn
        self._log_fn = log_fn
        self.state_key = state_key
        self.fields = fields
        self._trackers = {}

    def track_averages(self, engine=None, size_key=None):
        if (size_key is not None) and (not isinstance(engine, (bd.Engine, bd.State))):
            raise RuntimeError('Expected engine for provided size_key')
        self._trackers['average'] = AverageTracker(engine, size_key, self.fields)

    def reset_averages(self):
        self._trackers['average'].reset()

    def update_averages(self, value=None):
        value = self._get_value_fn() if value is None else value
        if value is None:
            return
        self._trackers['average'].update(value)

    def log_averages(self):
        value = self._trackers['average'].get()
        if value is None:
            return
        self._log_fn(value, tag='average')

    def log(self, value=None):
        value = self._get_value_fn() if value is None else value

        if value is None:
            return
        self._log_fn(value)


def _make_logger(lgr):
    if isinstance(lgr, GenericLogger):
        return lgr
    elif isinstance(lgr, Callable):
        return GenericLogger(lgr)
    else:
        raise RuntimeError(f'Loggers must be Callables, got: {type(lgr)}')


def _value_getter(self, state_key, fields):
    def get_value_fn():
        if state_key not in self:
            return None
        values = self[state_key]
        if values is None:
            return None
        keys = values.keys() if (fields is None) else fields
        ret = {key: values[key] for key in keys if key in values}
        if not ret:
            return None
        else:
            return ret

    return get_value_fn


class LoggerEngine(bd.Engine):
    def __init__(self):
        super().__init__()
        _create_state_dict_element(self, 'loggers')

    def attach_loggers(self, force=False):
        if force or (not self.loggers):
            if not hasattr(self, 'setup_loggers'):
                bd.warn(
                    'Could not find setup function for loggers. Will not attach to engine.'
                )
                return
            loggers = self.setup_loggers()
            if isinstance(loggers, Mapping):
                self.loggers = {
                    k: _make_logger(logger) for k, logger in loggers.items()
                }
            elif isinstance(loggers, Sequence):
                self.loggers = {
                    f'logger_{i}': _make_logger(logger)
                    for i, logger in enumerate(loggers)
                    if logger
                }
            else:
                raise RuntimeError('setup_loggers() must return a dictionary')

    # If fields is none then all will be logged
    def setup_tensorboard_logger(
        self,
        values_key,
        step_key,
        mode='scalar',
        category='plots',
        fields=None,
        directory='.',
        **kwargs,
    ):
        if 'tensorboard' not in self:
            directory = bd.process_path(directory, create=True)
            writer = SummaryWriter(directory, **kwargs)
            self.tensorboard = writer
        elif not isinstance(self.tensorboard, SummaryWriter):
            raise RuntimeError(
                f'self.tensorboard is not a SummaryWriter: {type(self.tensorboard)}'
            )
        else:
            writer = self.tensorboard

        category = category.rstrip('/') + '/'

        if mode in ['scalar', 'scalars']:
            get_value_fn = _value_getter(self, values_key, fields)

            def log_fn(values, tag=''):
                tag = f'_{tag}' if tag != '' else ''
                new_cat = category.rstrip('/') + tag.rstrip('/') + '/'
                if step_key not in self:
                    return
                step = self[step_key]
                for key, val in values.items():
                    writer.add_scalar(
                        f'{new_cat}{values_key}.{key}', val, global_step=step
                    )

        else:
            raise RuntimeError(f'Tensorboard logger mode "{mode}" not supported.')

        return GenericLogger(get_value_fn, log_fn, values_key, fields)

    def setup_tensorboard_logger_from_cfg(
        self,
        values_key,
        step_key,
        mode='scalar',
        category='plots',
        fields=None,
        cfg=None,
        **kwargs,
    ):
        cfg = _prepare_cfg(cfg, ['session_path'])
        directory = bd.process_path(os.path.join(cfg.session_path, 'tensorboard'))
        return self.setup_tensorboard_logger(
            values_key, step_key, mode, category, fields, directory, **kwargs
        )

    def setup_csv_logger(self, state_key, fields, directory='.', **kwargs):
        directory = bd.process_path(directory, create=True)
        if 'csv_writers' not in self:
            self.csv_writers = {}
        csv_writers = self.csv_writers

        def get_writer(tag):
            tag = f'_{tag}' if tag != '' else ''
            desc = f'{state_key.replace(".","_")}{tag}'
            writer = csv_writers.get(desc, None)
            if writer is None:
                writer = bd.CSVLogger(
                    desc, fields=fields, directory=directory, **kwargs
                )
                csv_writers[desc] = writer
            return writer

        get_value_fn = _value_getter(self, state_key, fields)

        def log_fn(values, tag=''):
            get_writer(tag)(values)

        return GenericLogger(get_value_fn, log_fn, state_key, fields)

    def setup_csv_logger_from_cfg(
        self, state_key, fields, cfg=None, delimiter=',', resume=True
    ):
        cfg = _prepare_cfg(cfg, ['session_path'])
        directory = bd.process_path(os.path.join(cfg.session_path, 'csv'), create=True)
        return self.setup_csv_logger(
            state_key,
            fields=fields,
            directory=directory,
            delimiter=delimiter,
            resume=resume,
        )

    def get_logging_dir_from_cfg(self, cfg=None):
        cfg = _prepare_cfg(cfg, ['session_path'])
        return bd.process_path(os.path.join(cfg.session_path, 'csv'), create=True)

    def log(self):
        for logger in self.loggers.values():
            logger.log()

    def log_averages(self):
        for logger in self.loggers.values():
            logger.log_averages()

    def update_averages(self):
        for logger in self.loggers.values():
            logger.update_averages()

    def reset_averages(self):
        for logger in self.loggers.values():
            logger.reset_averages()
