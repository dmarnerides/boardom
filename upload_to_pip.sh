#!/bin/bash

# Know what you are doing

rm -rf build/ dist/ boardom.egg*
python3 -m build 
python -m twine upload --repository pypi dist/*
