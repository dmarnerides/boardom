import uuid
from .handler import PersistentProcessPool
from .util import is_main_process, only_main_process

if is_main_process():
    _PROCESS_ID = uuid.uuid4().hex
else:
    _PROCESS_ID = None
