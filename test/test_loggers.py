import os
import shutil
import boardom as bd

_FILE = os.path.abspath(__file__)

_DIR = os.path.dirname(_FILE)

_ASSETS_DIR = os.path.join(_DIR, 'tmp_assets_csv_logger')


def setup_module():
    if os.path.exists(_ASSETS_DIR):
        shutil.rmtree(_ASSETS_DIR)
    os.mkdir(_ASSETS_DIR)


def teardown_module():
    if os.path.exists(_ASSETS_DIR):
        shutil.rmtree(_ASSETS_DIR)


class TestCSVLogger:
    def test_can_create_logger(self):
        bd.CSVLogger('foo', ('a', 'b', 'c'), directory=_ASSETS_DIR)
        assert os.path.exists(os.path.join(_ASSETS_DIR, 'foo.csv'))
