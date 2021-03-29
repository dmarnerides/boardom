import msgpack
from msgpack import ExtType
import zmq
import zmq.asyncio
import zlib
import pickle
import boardom as bd
from boardom.config.common import Group


def compress(obj, level=6):
    return zlib.compress(pack(obj), level)


def decompress(c_obj):
    return unpack(zlib.decompress(c_obj))


def msgpack_ext_pack(x):
    if isinstance(x, bd.Config):
        return ExtType(1, pickle.dumps(x))
    elif isinstance(x, Group):
        return ExtType(2, pickle.dumps(str(x)))
    elif isinstance(x, set):
        return ExtType(3, pickle.dumps(list(x)))
    return x


def msgpack_ext_unpack(code, data):
    if code == 1:
        return pickle.loads(data)
    elif code == 2:
        return Group(*pickle.loads(data).split(Group._SEPARATOR))
    elif code == 3:
        return set(pickle.loads(data))
    return msgpack.ExtType(code, data)


def pack(x):
    return msgpack.packb(x, default=msgpack_ext_pack)


def unpack(x):
    return msgpack.unpackb(x, ext_hook=msgpack_ext_unpack, raw=False)


class MsgpackSocket(zmq.Socket):
    def send_msgpack(self, obj, flags=0, copy=True, track=False):
        self.send(pack(obj), flags, copy=copy, track=track)

    def recv_msgpack(self, flags=0, copy=True, track=False):
        return unpack(self.recv(flags=flags, copy=copy, track=track))


class MsgpackContext(zmq.Context):
    _socket_class = MsgpackSocket


class MsgpackAsyncSocket(zmq.asyncio.Socket):
    async def send_msgpack(self, obj, flags=0, copy=True, track=False):
        return await self.send(pack(obj), flags, copy=copy, track=track)

    async def recv_msgpack(self, flags=0, copy=True, track=False):
        return unpack(await self.recv(flags=flags, copy=copy, track=track))


class MsgpackAsyncContext(zmq.asyncio.Context):
    _socket_class = MsgpackAsyncSocket
