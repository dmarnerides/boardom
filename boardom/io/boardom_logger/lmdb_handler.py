import os
import boardom as bd
import lmdb
from .serialization import pack, unpack


class LMDBHandler(metaclass=bd.Singleton):
    def __init__(self, directory, db_map_size=10485760, readonly=False):
        self.readonly = readonly
        directory = bd.process_path(directory, create=True)
        self.db_dirname = os.path.join(directory, 'lmdb_data')
        self.env = lmdb.open(self.db_dirname, subdir=True, readonly=self.readonly)

    def _write(self, key, data):
        if not isinstance(data, bytes):
            data = pack(data)
        with self.env.begin(write=True) as txn:
            txn.put(key.encode('utf-8'), data)

    def _read(self, key):
        with self.env.begin() as txn:
            val = txn.get(key.encode('utf-8'))
        if val:
            return unpack(val)
        else:
            return val

    def _create_key(self, *args):
        key = '/'.join([str(x) for x in args])
        count_key = f'{key}/count'
        count = self._read(count_key) or 0
        count += 1
        full_key = f'{key}/{count:012d}'
        return full_key, count_key, count

    def write_scalar(self, value, name, process_id):
        full_key, count_key, count = self._create_key(name, process_id)
        self._write(full_key, value)
        self._write(count_key, count)
