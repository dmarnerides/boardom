import os
from inspect import signature
from collections.abc import Mapping
import json
import torch
import boardom as bd
from .util import _create_state_dict_element, _prepare_cfg
from ..config.common import CHECKPOINT_KEYS


# Checkpointing state example:
#  checkpointing = {
#      'previous_checkpoint_num': 3,
#      'extension': '.pth',
#      'metafile': '/path/to/checkpoints/checkpoints.json',
#      'metadata': {
#          'k1' : {
#              'state_key': 'state.key_1',
#              'directory': '/path/to/checkpoints/',
#              'overwrite': True,
#              'use_timestamps': True,
#              'save_state_dicts': True,
#              'files': {
#                  'k1_chk1': {
#                      'timestamp': '2020-12-31T23:59:59',
#                      'basename': 'state.key_1.2020-12-31T23:59:59.tag.pth',
#                      'extra': None,
#                  },
#                  'k1_chk2': {
#                      'timestamp': '2021-01-01T23:59:59',
#                      'basename': 'state.key_1.2021-01-01T23:59:59.tag.pth',
#                      'extra': None,
#                  },
#                  'k1_chk3': {
#                        ...
#                  },
#              },
#          },
#          ...
#  }
#

EXT = '.pth'


class Checkpoint(bd.Engine):
    def __init__(self):
        super().__init__()
        _create_state_dict_element(self, 'checkpointing')

    # If state_keys are provided they override the get_checkpoint_settings() call
    def attach_checkpointers(
        self,
        *state_keys,
        directory='checkpoints',
        overwrite=True,
        use_timestamps=True,
        save_state_dicts=True,
        sanitize_metadata=False,
    ):
        def create_default_dict():
            return {
                'state_key': bd.Null,
                'directory': bd.Null,
                'overwrite': overwrite,
                'use_timestamps': use_timestamps,
                'save_state_dicts': save_state_dicts,
                'files': {},
            }

        directory = bd.process_path(directory, create=True)
        metafile = os.path.join(directory, 'checkpoints.json')

        metadict = {}
        if state_keys:
            chkp_data = [dict(state_key=sd) for sd in state_keys]
        else:
            chkp_data = self.event('get_checkpoint_settings')
        for datum in chkp_data:
            if datum is None:
                continue
            if isinstance(datum, dict):
                datum = _create_metadict(datum, directory, create_default_dict())
                altered_key = datum['state_key'].replace('.', '_')
                if altered_key in metadict:
                    msg = (
                        f'Checkpoint settings for {datum["state_key"]} already defined'
                    )
                    raise RuntimeError(msg)
                metadict[altered_key] = datum
            elif isinstance(datum, (list, tuple)):
                for subdatum in datum:
                    if not isinstance(subdatum, dict):
                        msg = f'Invalid checkpoint member : {subdatum}'
                        raise RuntimeError(msg)
                    subdatum = _create_metadict(
                        subdatum, directory, create_default_dict()
                    )
                    altered_key = subdatum['state_key'].replace('.', '_')
                    if altered_key in metadict:
                        msg = f'Checkpoint settings for {subdatum["state_key"]} already defined'
                        raise RuntimeError(msg)
                    metadict[altered_key] = subdatum
            else:
                raise RuntimeError(f'Invalid checkpoint member : {datum}')
        self.checkpointing = {
            'extension': EXT,
            'previous_checkpoint_num': 0,
            'metafile': metafile,
            'metadata': metadict,
        }

        if os.path.exists(metafile):
            bd.log('Found previous checkpoints. Loading metadata...')
            with open(metafile, 'r') as f:
                self.checkpointing.update(json.load(f))
            bd.log('Done loading metadata.')
        if sanitize_metadata:
            _sanitize(self.checkpointing)

    def attach_checkpointers_from_cfg(
        self,
        *state_keys,
        cfg=None,
        sanitize_metadata=False,
    ):
        cfg = _prepare_cfg(cfg, CHECKPOINT_KEYS + ['session_path'])
        directory = os.path.join(cfg.dg.session_path, 'checkpoints')
        self.attach_checkpointers(
            *state_keys,
            directory=directory,
            overwrite=cfg.overwrite,
            use_timestamps=cfg.use_timestamps,
            save_state_dicts=cfg.save_state_dicts,
            sanitize_metadata=sanitize_metadata,
        )
        # Rebase the directory from metadata in case it is accessed
        # from a different mount point
        for val in self.checkpointing.metadata.values():
            val.directory = directory

    def save_checkpoint(
        self,
        *state_keys,
        extra_meta=None,
        tag=None,
        save_fn=torch.save,
        force_no_overwrite=False,  # Helpful for saving "special" checkpoints
    ):
        with bd.interrupt_guard(reason='Saving checkpoints'):
            _save(
                self,
                state_keys,
                extra_meta=extra_meta,
                tag=tag,
                save_fn=save_fn,
                force_no_overwrite=force_no_overwrite,
            )

    def load_latest(
        self, *keys, exclude=None, load_fn=torch.load, strict=True, **kwargs
    ):
        chkp = self.checkpointing
        chkp_num = chkp.previous_checkpoint_num
        if chkp_num == 0:
            return
        for altered_key, val in chkp.metadata.items():
            directory = val.directory
            state_key = val.state_key
            if keys and (state_key not in keys):
                continue
            if (exclude is not None) and (state_key in exclude):
                continue
            current_key = f'{altered_key}_chk{chkp_num}'
            if current_key not in val.files:
                bd.warn(
                    f'Could not find file for "{state_key}" latest checkpoint. Skipping...'
                )
                continue
            latest = val.files[current_key]
            basename = latest.basename
            full_name = os.path.join(directory, basename)
            loaded = load_fn(full_name, **kwargs)
            _set_state(self, state_key, loaded, val.save_state_dicts, strict=strict)
            bd.log(f'Loaded {state_key} checkpoint: {basename}')


def _set_state(self, state_key, loaded, save_state_dicts, strict=True):
    obj = self.get(state_key, bd.Null)
    if obj is bd.Null:
        self[state_key] = loaded
    elif save_state_dicts and hasattr(obj, 'load_state_dict'):
        load_state_dict = obj.load_state_dict
        sig = signature(load_state_dict)
        if 'strict' in sig.parameters:
            load_state_dict(loaded, strict=strict)
        else:
            load_state_dict(loaded)
    elif (
        save_state_dicts
        and (not hasattr(obj, 'load_state_dict'))
        and isinstance(obj, Mapping)
        and isinstance(loaded, Mapping)
    ):
        for subkey, subloaded in loaded.items():
            new_key = f'{state_key}.{subkey}'
            _set_state(self, new_key, subloaded, save_state_dicts, strict=strict)
        pass
    else:
        self[state_key] = loaded


def _get_state(obj, save_state_dicts):
    if isinstance(obj, bd.State):
        return {key: _get_state(val, save_state_dicts) for key, val in obj.items()}
    elif (
        save_state_dicts
        and hasattr(obj, 'state_dict')
        and callable(obj.state_dict)
        and hasattr(obj, 'load_state_dict')
        and callable(obj.load_state_dict)
    ):
        return obj.state_dict()
    else:
        return obj


def _create_metadict(given_dict, directory, new_dict):
    valid_keys = set(new_dict.keys())
    directory = given_dict.get('directory', directory)
    given_dict['directory'] = bd.process_path(directory, create=True)
    new_dict.update(given_dict)
    for key, val in new_dict.items():
        if key not in valid_keys:
            valid = ", ".join(valid_keys)
            raise RuntimeError(
                f'Invalid key for checkpointer setup: {key}' f'\nValid keys: {valid}'
            )
        if val is bd.Null:
            raise RuntimeError(f'Checkpointer setup missing "{key}" value')
    return new_dict


def _sanitize(chkp):
    meta = chkp.metadata
    for key, val in meta.items():
        directory = val.directory
        for subkey, subval in val.files.items():
            full_path = os.path.join(directory, subval.basename)
            if not os.path.exists(full_path):
                del val.files[subkey]


def _file_meta(timestamp, basename, checkpoint_id):
    return {
        'timestamp': timestamp,
        'basename': basename,
        'checkpoint_id': 1,
        'extra': None,
    }


def _save(self, state_keys, extra_meta, tag, save_fn, force_no_overwrite):
    chkp = self.checkpointing
    previous_checkpoint_num = chkp.previous_checkpoint_num
    current_checkpoint_num = previous_checkpoint_num + 1
    all_meta = chkp.metadata
    state_keys = state_keys or [x['state_key'] for x in all_meta.values()]
    timestamp = bd.timestamp()
    for state_key in state_keys:
        altered_key = state_key.replace('.', '_')
        metadata = all_meta[altered_key]
        overwrite = metadata.overwrite and (not force_no_overwrite)
        directory = metadata.directory
        if state_key not in self:
            raise RuntimeError(f'Could not find {state_key} to checkpoint.')
        to_save = _get_state(
            self[state_key], save_state_dicts=metadata.save_state_dicts
        )
        new_tag = f'.{str(tag)}' if tag else ''
        str_timestamp = f'.{timestamp}' if metadata.use_timestamps else ''
        basename = f'{state_key}{str_timestamp}{new_tag}{chkp.extension}'
        # if not overwriting, make sure that we give a new name to the file
        # (incase it already exists)
        if not overwrite:
            _fullname = os.path.join(directory, basename)
            basename = os.path.basename(bd.number_file_if_exists(_fullname))
        full_name = os.path.join(directory, basename)
        safe_save(to_save, full_name, save_fn)

        # If we are overwiting and the new name is not the same
        # as the previous one delete the previous one
        if overwrite:
            prev_key = f'{altered_key}_chk{previous_checkpoint_num}'
            prev_meta = metadata.files.get(prev_key, None)
            if prev_meta is not None:
                prev_fullname = os.path.join(directory, prev_meta.basename)
                if full_name != prev_fullname:
                    os.remove(prev_fullname)
                del metadata.files[prev_key]

        # TODO: Add metadata info
        metakey = f'{altered_key}_chk{current_checkpoint_num}'
        metadata.files[metakey] = {
            'timestamp': timestamp,
            'basename': basename,
            'extra': extra_meta,
        }
        bd.log(f'Saved {state_key} checkpoint: {basename}')

    chkp.previous_checkpoint_num = current_checkpoint_num
    # Save the metadata too
    safe_save(chkp, chkp.metafile, json_save_func)


# Save func is save(object, filename)
def safe_save(obj, full_name, save_func):
    if os.path.exists(full_name):
        # Write temporary new file
        tmp_file = full_name + '.tmp'
        save_func(obj, tmp_file)
        # Backup current file
        bak_current_file = full_name + '.bak'
        os.replace(full_name, bak_current_file)
        # Rename temporary file
        os.replace(tmp_file, full_name)
        # Remove backup
        os.remove(bak_current_file)
    else:
        save_func(obj, full_name)


def json_save_func(obj, file):
    with open(file, 'w') as f:
        json.dump(obj, f, indent=4, sort_keys=True)
