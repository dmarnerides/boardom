import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_PATH = os.path.abspath(os.path.join(THIS_DIR, '..', 'dist'))
INDEX_PATH = os.path.join(DIST_PATH, 'index.html')
STATICS = ['index.bundle.js', 'favicon.ico']
SOCKET_URLS = ['front_socket', 'engine_socket']
ALL_FILES = STATICS + SOCKET_URLS

DOTDIR = os.path.expanduser('~/.boardom')
DOTFILE = os.path.join(DOTDIR, 'settings.json')
DEFAULT_DOTFILE_DATA = {'base_dirs': ['~/boardom_sessions']}
BD_FILENAME = '.bdsession'


def deep_update(d1, d2):
    for key, val in d1.items():
        if isinstance(val, (int, float, str, bool)):
            d1[key] = d2.get(key, val)
        elif isinstance(val, list):
            d1[key] += d2.get(key, [])
        elif isinstance(val, dict):
            d1[key] = deep_update(val, d2.get(key, {}))
    return d1
