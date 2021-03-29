#!/usr/bin/env python

from multiprocessing import Process
import time
import os

CWD = os.path.abspath(os.path.curdir)


def front():
    os.system(f'cd {CWD}; ./front.sh')


def server():
    os.system(f'cd {CWD}; ./server.sh')


Process(target=front, daemon=True).start()
Process(target=server, daemon=True).start()

try:
    while True:
        time.sleep(0.1)
except:
    exit()
