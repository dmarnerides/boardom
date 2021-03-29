import atexit
import time
import zmq
from threading import Thread, Lock
from multiprocessing import Process, Value
from .subprocess import _LoggerSubprocess
from .serialization import MsgpackContext
from ...multiprocessing import _PROCESS_ID
import boardom as bd

# TODO: CHECK WE ARE IN MASTER PROCESS WHEN SPAWNING CHILD
# TODO: MAKE SURE NONE OF THE PORTS ARE THE ONE WE USE FOR THE SERVER
# TODO: Also exclude other ports, like tensorboard and visdom +1,2,3
# TODO: Check correct handling of default_grp (e.g. when used in config files)


# This is functionality to handle requests from the subprocess
class _SubprocessRequestsMixin:
    def _handle_request_cfg_store(self, task):
        bd.log('Got Request to send config store.')
        if bd.cfg._prv['done_setup']:
            self._send_cfg_full()
        else:
            bd.log('Config store not sent. (not setup yet).')

    def _handle_default(self, task):
        bd.log(f'I do not know how to handle {task["type"]}')


# This is for private API functions
class _PrivateAPIMixin:
    # This should only be called after bd.cfg.setup()
    def _send_cfg_full(self):
        bd.log('Sending config store.')
        self.to_child.send_msgpack(
            {'type': 'ENGINE_CFG_FULL', 'payload': bd.cfg._get_data_dict()}
        )

    # This should only be called after bd.cfg.setup()
    def _start_lmdb(self):
        bd.log('Starting LMDB.')
        payload = {'session_path': bd.cfg.session_path}
        self.to_child.send_msgpack({'type': 'START_LMDB', 'payload': payload})

    # This is used by Config.set
    def _update_cfg_value(self, arg_name, group, value):
        self.to_child.send_msgpack(
            {
                'type': 'SET_CFG_VALUE',
                'payload': {'arg_name': arg_name, 'value': value, 'group': group},
            }
        )


# This is user facing API
class _APIMixin:
    def plot_xy(self, x, y, name):
        # This is further processed by subprocess
        self.to_child.send_msgpack(
            {
                'type': 'PLOT_XY_SCATTER',
                'payload': {
                    'x': x,
                    'y': y,
                    'name': name,
                    'time': int(time.time() * 1000),
                },
            }
        )


def _shutdown():
    BoardomLogger()._exit()


class _NullChild:
    def __getattr__(self, *args, **kwargs):
        raise RuntimeError(
            'Must call bd.boardom_logger.start() before using the Boardom Logger.'
        )


class BoardomLogger(
    _SubprocessRequestsMixin, _APIMixin, _PrivateAPIMixin, metaclass=bd.Singleton
):
    _started = False

    def __init__(self):
        self.to_child = _NullChild()

    def start(self):
        if not BoardomLogger._started:
            BoardomLogger._started = True
            self._started = True
            ctx = MsgpackContext()
            self.to_child = ctx.socket(zmq.PAIR)
            self.to_child.bind('tcp://*:0')
            self.parent_port = int(
                self.to_child.getsockopt(zmq.LAST_ENDPOINT)
                .decode('utf-8')
                .split(':')[-1]
            )
            self.connected = Value('i', 0)
            self.exited = Value('i', 0)
            self.child_port = Value('i', 0)
            self.process = Process(
                target=_LoggerSubprocess,
                daemon=True,
                args=(
                    self.connected,
                    self.exited,
                    self.child_port,
                    self.parent_port,
                    'http://localhost:8089',
                    2,
                    _PROCESS_ID,
                ),
            )
            # TODO: MAKE THESE STATEMENTS WITHSTAND KEYBOARDINTERRUPTS
            # and SIGINT
            self.process.start()
            atexit.register(_shutdown)
            self.from_child = ctx.socket(zmq.PAIR)
            # wait for port allocation from child and then connect
            while (self.child_port.value == 0) and self.process.is_alive():
                time.sleep(0.005)
            self.from_child.connect(f'tcp://127.0.0.1:{self.child_port.value}')
            self._task_lock = Lock()
            self._queued_tasks = []
            self._child_watcher_sleep_time = 0.01
            thr = Thread(target=self._watch_child, daemon=True)
            thr.start()
            if bd.cfg._prv['done_setup']:
                self._send_cfg_full()
                self._start_lmdb()
        return self

    def block_until_connected(self, t_check_connect=0.01, block_timeout=-1):
        if self.connected.value == 0:
            print('Connecting to Boardom (blocking)...')
        else:
            print('Boardom already connected.')
        start = time.time()
        timed_out = False
        while (self.connected.value == 0) and self.process.is_alive():
            time.sleep(t_check_connect)
            if (block_timeout > 0) and ((time.time() - start) > block_timeout):
                timed_out = True
                break
        if self.connected.value == 0:
            if self.process.is_alive():
                self.process.join()
            if timed_out:
                print('Boardom connection timed out')
            print('Boardom not connected')
        else:
            print('Boardom connected, unblocking...')

    def _exit(self):
        self.to_child.send_msgpack({'type': 'EXIT'})
        while (self.exited.value == 0) and self.process.is_alive():
            time.sleep(0.01)

    def _check_and_do_task(self, task, do_passive, do_synced):
        if (task['meta'].get('passive', False) and do_passive) or (
            task['meta'].get('needs_sync', False) and do_synced
        ):
            try:
                handler = getattr(self, f'_handle_{task["type"].lower()}')
            except AttributeError:
                handler = self._handle_default
            handler(task)
            return True
        else:
            return False

    def _do_tasks(self, do_passive, do_synced):
        # First try and do earlier tasks
        self._task_lock.acquire()
        should_release = True
        self._queued_tasks = [
            x
            for x in self._queued_tasks
            if not self._check_and_do_task(x, do_passive, do_synced)
        ]

        while True:
            try:
                task = self.from_child.recv_msgpack(zmq.NOBLOCK)
            except zmq.ZMQError:
                self._task_lock.release()
                should_release = False
                break
            else:
                handled = self._check_and_do_task(task, do_passive, do_synced)
                if not handled:
                    self._queued_tasks.insert(task, 0)
        if should_release:
            self._task_lock.release()

    def _watch_child(self):
        while True:
            self._do_tasks(do_passive=True, do_synced=False)
            time.sleep(self._child_watcher_sleep_time)

    def synchronize(self):
        self._do_tasks(self, do_passive=False, do_synced=True)

    def synchronize_all(self):
        self._do_tasks(self, do_passive=True, do_synced=True)
