import os
from collections.abc import Callable
import boardom as bd
from ..multiprocessing import _PROCESS_ID
from .git import maybe_autocommit
from boardom.board.server.common import BD_FILENAME
from .common import _create_datum, Group


def _get(cfg, arg):
    return cfg._prv['data'][arg][Group()]['value']


def _set(cfg, arg, value):
    cfg._prv['data'][arg][Group()]['value'] = value


class Session:
    def __init__(self, stream_replicator=None):
        self.stream_replicator = stream_replicator

    def deactivate(self):
        if self.stream_replicator is not None:
            self.stream_replicator.deactivate()


def _create_session(cfg, session_name=None):
    cfg.setup()
    if not cfg._prv['has_core_config']:
        raise RuntimeError('Can not create_session without core configuration')
    session = Session()

    # Configure session name
    if isinstance(session_name, str):
        _set(cfg, 'session_name', session_name)
    elif isinstance(session_name, Callable):
        session_name = session_name(cfg)
        _set(cfg, 'session_name', session_name)
    elif session_name is None:
        session_name = _get(cfg, 'session_name')
    else:
        raise RuntimeError(
            f'Unknown type for session_name parameter: {type(session_name)}'
        )
    bd.log(f'Creating {session_name} session.')

    project_path = bd.process_path(_get(cfg, 'project_path'), create=True)
    bd.log(f'Project path: {project_path}.')

    session_path = os.path.join(project_path, session_name)
    bd.make_dir(session_path)

    boardom_path = bd.make_dir(os.path.join(session_path, '.boardom'))
    session_file = os.path.join(boardom_path, BD_FILENAME)
    # TODO: Improve Management of Session Files
    #     -- Maybe use a single file?
    #     -- Maybe add information
    if not os.path.exists(session_file):
        with open(session_file, 'w') as f:
            f.write('42')

    # Maybe create log
    create_log = _get(cfg, 'log_stdout')
    if create_log:
        log_name = f'{session_name}_{_PROCESS_ID}.log'
        logdir = os.path.join(session_path, 'log')
        logdir = bd.process_path(logdir, create=True)
        logfile = os.path.join(logdir, log_name)
        logfile = bd.number_file_if_exists(logfile)
        bd.log(f'Creating log file at {logfile}')
        session.stream_replicator = bd.replicate_std_stream(logfile, 'stdout')

    # Maybe copy config files
    cfg_files = cfg._prv['cfg_files']
    copy_config_files = _get(cfg, 'copy_config_files')
    if copy_config_files:
        for i, filename in enumerate(cfg_files):
            config_path = os.path.join(session_path, 'cfg')
            bd.make_dir(config_path)
            if i == 0:
                bd.log(f'Copying configuration files to {config_path}')
            fname, ext = os.path.splitext(filename)
            copied_config_filename = f'{fname}_{_PROCESS_ID}{ext}'
            bd.copy_file_to_dir(
                filename,
                config_path,
                number=True,
                new_name=copied_config_filename,
            )

    # Maybe save full config
    save_full_config = _get(cfg, 'save_full_config')
    if save_full_config:
        config_path = os.path.join(session_path, 'cfg')
        bd.make_dir(config_path)
        config_file = os.path.join(config_path, f'full_cfg_{_PROCESS_ID}.bd')
        config_file = bd.number_file_if_exists(config_file)
        bd.log(f'Saving full configuration at: {config_file}')

        # Makes an entry for the saved settings file
        def _make_entry(key, val):
            if any(isinstance(val, x) for x in [list, tuple]):
                val = ' '.join([str(x) for x in val])
            return f'{key} {str(val)}'

        args_to_print = [_make_entry(key, val) for key, val in cfg.__dict__.items()]
        args_to_print.sort()
        bd.write_string_to_file('\n'.join(args_to_print), config_file)

    autocommit = _get(cfg, 'autocommit')
    only_run_same_hash = _get(cfg, 'only_run_same_hash')
    _, _, autohash = maybe_autocommit(autocommit, only_run_same_hash, session_path)
    pid_fname = f'process.{_PROCESS_ID}'
    if autohash is not None:
        pid_fname += f'.{autohash}'

    #  process_dir = bd.make_dir(os.path.join(boardom_path, 'processes'))
    #  process_id_file = os.path.join(process_dir, pid_fname)
    #
    #  if os.path.exists(process_id_file):
    #      raise RuntimeError(
    #          'Process File Already Exists?!? That is unlucky. Please run again..'
    #          f'\n id: {process_id_file}'
    #      )
    #  else:
    #      with open(process_id_file, 'w') as f:
    #          f.write('42')
    if _get(cfg, 'print_cfg'):
        bd.write('-' * 80)
        bd.write(cfg)
        bd.write('-' * 80)

    cfg._prv['data']['session_path'] = _create_datum(session_path)
    return session
