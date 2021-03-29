import os
import json
import boardom
from .common import BD_FILENAME, DEFAULT_DOTFILE_DATA, DOTFILE, deep_update
from .data_mixin import _DataMixin


def is_subdir(path, paths):
    path = path + '/'
    for x in paths:
        x = x + '/'
        if x == path:
            continue
        if path.startswith(x):
            return True
    return False


class _ProcessMixin:
    def initialize_process_infos(self):
        process_infos = {}
        for path in self.store['valid_paths']:
            p_ids = [
                fname.split('.')[2]
                for fname in os.listdir(os.path.join(path, '.boardom'))
                if fname.startswith('.process')
            ]
            process_infos.update(
                {pid: {'path': path, 'active': False} for pid in p_ids}
            )
        self.store['processes'] = process_infos

    # This is only the process_id for a newly connected process
    def add_new_process(self, process_id):
        if process_id in self.store['processes']:
            self.store['processes'][process_id]['active'] = True
        else:
            self.store['processes'][process_id] = {
                'path': None,
                'active': True,
            }

    def deactivate_process(self, process_id):
        self.store['processes'][process_id]['active'] = False

    def get_process_info(self, process_id):
        return {'id': process_id, **self.store['processes'][process_id]}

    def get_all_processes(self):
        return list(
            {'id': key, **val} for key, val in datastore.store['processes'].items()
        )


class _ConfigMixin:
    def add_cfg_store(self, store, process_id):
        cfg_store = self.store['cfg']
        for arg_name, grp_dict in store.items():
            for group, grp_dict in grp_dict.items():
                cfg_id = str(hash((arg_name, group, process_id)))

                cfg_store[cfg_id] = {
                    'name': arg_name,
                    'group': group,
                    'value': grp_dict['value'],
                    'tags': grp_dict.get('tags', []),
                    'meta': grp_dict.get('meta', {}),
                    'process_id': process_id,
                    'id': cfg_id,
                }

    def set_cfg_value(self, payload, process_id):
        name, group, value = payload['arg_name'], payload['group'], payload['value']
        cfg_id = str(hash((name, group, process_id)))
        cfg_store = self.store['cfg']
        if cfg_id not in cfg_store:
            print(
                f'[Server] Could not find cfg {name} for '
                f'group {group} and process_id {process_id}'
            )
            return None
        else:
            cfg_store[cfg_id]['value'] = value
            return cfg_store[cfg_id]

    def get_cfg_store(self, process_id=None):
        if process_id:
            return {
                k: v
                for k, v in self.store['cfg'].items()
                if v['process_id'] == process_id
            }
        else:
            return self.store['cfg']


class _DataStore(_DataMixin, _ProcessMixin, _ConfigMixin):
    def __init__(self):
        self._initialized = False

    # TODO: Convert most of these to async operations and move to the initialize_function to not block startup
    def create(self):
        if self._initialized:
            return self.store
        self.store = {
            'cfg': {},
            'valid_paths': None,
            'data': {'ids': {}},
            'visualisations': {'ids': {}},
            'processes': {},
            'user_settings': DEFAULT_DOTFILE_DATA,
        }
        self.read_user_settings()
        self.validate_paths()
        self.check_valid_session_dirs()
        self.initialize_process_infos()
        self._initialized = True
        return self.store

    # TODO: MAKE THESE ASYNC
    async def initialize(self):
        #  await self.load_existing_data()
        pass

    def read_user_settings(self):
        if os.path.isfile(DOTFILE):
            with open(DOTFILE) as json_file:
                new_data = json.load(json_file)
            deep_update(self.store['user_settings'], new_data)

    def validate_paths(self):
        stgs = self.store['user_settings']
        paths = stgs['base_dirs']
        paths = list(set([boardom.process_path(path) for path in paths]))
        stgs['base_dirs'] = [path for path in paths if not is_subdir(path, paths)]

    def check_valid_session_dirs(self):
        base_dirs = self.store['user_settings']['base_dirs']
        self.store['valid_paths'] = [
            os.path.join(root_path, path)
            for root_path in base_dirs
            for path, _, files in os.walk(root_path)
            if BD_FILENAME in os.walk(os.path.join(root_path, path))[2]
        ]


datastore = _DataStore()
