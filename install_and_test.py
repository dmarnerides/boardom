#!/usr/bin/env python
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('tests', nargs='?', default='test')
parser.add_argument('--notest', action='store_true')
parser.add_argument('--noinstall', action='store_true')
args = parser.parse_args()

# use this for board
# npm run build

if not args.noinstall:
    os.system('rm -rf build boardom.egg-info')
    os.system('python setup.py install')
if not args.notest:
    os.system(f'pytest {args.tests}')
