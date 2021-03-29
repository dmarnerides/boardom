from contextlib import contextmanager
import os
import portalocker
from os import path
import sys
from shutil import copyfile
import errno

__all__ = [
    'split_path',
    'make_dir',
    'copy_file_to_dir',
    'process_path',
    'write_string_to_file',
    'number_file_if_exists',
]


def main_file_path():
    main_module = sys.modules['__main__']
    if hasattr(main_module, '__file__'):
        filepath = os.path.abspath(main_module.__file__)
        return os.path.dirname(filepath)
    else:
        return os.getcwd()


def split_path(directory):
    """Splits a full filename path into its directory path, name and extension

    Args:
        directory (str): Directory to split.

    Returns:
        tuple: (Directory name, filename, extension)
    """
    directory = process_path(directory)
    name, ext = path.splitext(path.basename(directory))
    return path.dirname(directory), name, ext


def make_dir(directory):
    """Make a new directory. Calls os.makedirs but suppresses path exists error.

    Args:
        directory (str): Directory to make.
    """
    directory = process_path(directory, False)
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return directory


def copy_file_to_dir(file, directory, number=False, new_name=None):
    """Copies a file to a directory

    Args:
        file (str): File to copy.
        directory (str): Directory to copy file to.
    """
    new_name = new_name or file
    new_name = os.path.basename(new_name)
    file_path = path.join(directory, new_name)
    file_path = process_path(file_path)
    if number:
        file_path = number_file_if_exists(file_path)
    copyfile(file, file_path)
    return file_path


def process_path(directory, create=False):
    """Expands home path, finds absolute path and creates directory (if create is True).

    Args:
        directory (str): Directory to process.
        create (bool, optional): If True, it creates the directory.

    Returns:
        str: The processed directory.
    """
    directory = path.expanduser(directory)
    directory = path.normpath(directory)
    directory = path.abspath(directory)
    directory = path.realpath(directory)
    if create:
        make_dir(directory)
    return directory


def write_string_to_file(contents, filename, directory=".", append=False):
    """Writes contents to file.

    Args:
        contents (str): Contents to write to file.
        filename (str): File to write contents to.
        directory (str, optional): Directory to put file in.
        append (bool, optional): If True and file exists, it appends contents.

    Returns:
        str: Full path to file.
    """
    full_name = path.join(process_path(directory), filename)
    mode = "a" if append else "w"
    with open(full_name, mode) as file_handle:
        file_handle.write(contents)
    return full_name


def number_file_if_exists(full_filename):
    full_filename = process_path(full_filename)
    path, basename = os.path.split(full_filename)
    filename, ext = os.path.splitext(basename)
    current = full_filename
    count = 0
    while os.path.isfile(current):
        count += 1
        current = os.path.join(path, f'{filename}_{count}{ext}')
    return current


@contextmanager
def filelock(filename=None):
    if filename is None:
        filename = os.path.join(main_file_path(), '.bd.lock')
    print('Acquiring lock...')
    with portalocker.Lock(filename, 'w', timeout=60) as lockfile:
        lockfile.flush()
        os.fsync(lockfile.fileno())
        yield
        if os.path.exists(filename):
            os.remove(filename)
