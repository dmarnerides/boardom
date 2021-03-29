import os
import subprocess
import time
import boardom as bd
from .gitignore import IGNORE, IGNORE_FIRST

# TODO: (SOS) Guard git calls to ignore CTRL-C and SIGINTS until finished
# TODO: (SOS) Make it only master process
# TODO: Add logging
# TODO: Manage gitignore (e.g. .pth files, boardom files etc)
# TODO: Smart adding of files to index (?)


class _Gitter:
    BD_BRANCH_NAME = 'bd_autocommits'

    def __init__(self):
        self.directory = bd.process_path(bd.main_file_path())
        self.lockfile = os.path.join(self.directory, '.bd.gitlock')
        if not os.path.isdir(self.directory):
            raise RuntimeError(f'[boardom] Invalid directory {self.directory}')
        self.is_git_dir = None
        self.status = None
        self.status_codes = None
        self.status_files = None
        self.current_branch = None
        self.has_branches = None
        self.has_unstaged = None

    def __call__(self, cmd, should_raise=True, log=False):
        try:
            if log:
                bd.log(f'git {cmd}')
            proc = subprocess.run(
                f'cd {self.directory}; git {cmd}',
                shell=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            if should_raise:
                raise
            pass
        success = proc.returncode == 0
        return proc, success

    def _check_is_git_dir(self):
        _, success = self('rev-parse', should_raise=False)
        return success

    def _set_base_git_dir(self):
        proc, _ = self('rev-parse --git-dir')
        gitdir = proc.stdout.rstrip()
        if os.path.isabs(gitdir):
            self.directory = os.path.dirname(gitdir)
        elif gitdir != '.git':
            raise RuntimeError(
                f'[boardom] Could not get base git directory for {self.directory}.'
            )

    def check_current_branch_name(self):
        proc, has_branches = self(
            'rev-parse --abbrev-ref --symbolic-full-name HEAD', should_raise=False
        )
        current_branch = proc.stdout.rstrip()
        return current_branch, has_branches

    def _get_current_commit_hash(self):
        proc, _ = self('rev-parse HEAD')
        return proc.stdout.rstrip()

    def _update_status(self):
        self.is_git_dir = self._check_is_git_dir()
        if not self.is_git_dir:
            return
        self._set_base_git_dir()
        self.current_branch, self.has_branches = self.check_current_branch_name()
        self.current_commit = self._get_current_commit_hash()

        # Set status
        proc, _ = self('status --porcelain ')
        status = proc.stdout.split('\n')
        self.status = [[y for y in x.split(' ') if y] for x in status if x]

        #  self.untracked = [x[1] for x in self.status if x[0] == '??']
        #  self.modified = [x[1] for x in self.status if x[0] == 'M']
        #  self.added = [x[1] for x in self.status if x[0] == 'A']
        #  self.deleted = [x[1] for x in self.status if x[0] == 'D']
        #  self.renamed = [x[1] for x in self.status if x[0] == 'R']
        #  self.copied = [x[1] for x in self.status if x[0] == 'C']

    def _print_info(self):
        print(f'Is git dir: {self.is_git_dir}')
        print(f'Directory: {self.directory}')
        print(f'Current branch: {self.current_branch}')
        print(f'Has branches: {self.has_branches}')
        print(f'Status: {self.status}')

    def _message(self):
        timestamp = time.strftime("Date: %Y/%m/%d, Time: %H:%M:%S")
        return f'Boardom autocommit\n{timestamp}'

    def _initialize_and_commit(self):
        proc, _ = self('init')
        self._update_status()
        self._add_gitignore()
        if not self.status:
            raise RuntimeError(
                f'[boardom] Git autocommit failed (no files in directory {self.directory})'
            )
        self(f'checkout -b {self.BD_BRANCH_NAME}', log=True)
        self('add .', log=True)
        self(f'commit -m "{self._message()}"', log=True)
        self("checkout -b master", log=True)
        print('[boardom] Initialized git dir and made commit.')
        commit_hash = self._get_current_commit_hash()
        return commit_hash, 'master', commit_hash

    def _do_parallel_commit(self):
        self._update_status()
        self._add_gitignore()
        prev_commit = self.current_commit
        prev_branch = self.current_branch
        self('add .', log=True)
        self('stash', log=True)
        proc, success = self(f'checkout -b {self.BD_BRANCH_NAME}', log=True)
        if not success:
            self(f'checkout {self.BD_BRANCH_NAME}')
        self(f'git merge {prev_branch} --no-commit -s recursive -Xtheirs', log=True)
        self('checkout stash -- .', log=True)
        self('add .', log=True)
        self(f'commit -m "{self._message()}"', log=True)
        new_commit = self._get_current_commit_hash()
        self(f'checkout {prev_branch}', log=True)
        self('stash pop', log=True)
        print(f'[boardom] Made autocommit: {new_commit}')
        return prev_commit, prev_branch, new_commit

    def _add_gitignore(self):
        # Add gitignore if flagged and no gitignore exists
        ignore_file = os.path.join(self.directory, '.gitignore')
        if not os.path.exists(ignore_file):
            with open(ignore_file, 'w') as f:
                f.write(IGNORE)
            print('[boardom] Created .gitignore')
        else:
            with open(ignore_file) as f:
                first_line = f.readline()
            if first_line != IGNORE_FIRST:
                with open(ignore_file, 'r+') as f:
                    content = f.read()
                    f.seek(0, 0)
                    f.write(IGNORE + content)
                print(f'[boardom] Prepended to .gitignore.')
            else:
                print(f'[boardom] .gitignore already correct.')

    def _autocommit(self):
        self._update_status()
        # If git is not initialized, run git init
        if not self.is_git_dir:
            return self._initialize_and_commit()
        else:
            # If we are on master branch: (This should be the expected situation)
            if self.current_branch == 'HEAD':
                raise RuntimeError(
                    '[boardom] Git is in a detached head state. Please checkout a branch.'
                )
            elif self.current_branch == self.BD_BRANCH_NAME:
                raise RuntimeError(
                    '[boardom] Current branch is autocommit branch. '
                    'If changes were made, please use git stash and '
                    'git stash pop on your main branch to keep your changes.'
                )
            return self._do_parallel_commit()

    def autocommit(self):
        self._update_status()
        with bd.filelock(self.lockfile):
            with bd.interrupt_guard():
                return self._autocommit()


def maybe_autocommit(autocommit, only_run_same_hash, session_path):
    if autocommit:
        git = _Gitter()
        current_hash, current_branch, autohash = git.autocommit()
        if only_run_same_hash:
            git_hash_file = os.path.join(session_path, '.githash_guard')
            if not os.path.isfile(git_hash_file):
                bd.write_string_to_file(autohash, git_hash_file)
            else:
                with open(git_hash_file, 'r') as f:
                    old_git_hash = f.read()
                if old_git_hash != autohash:
                    bd.error(
                        'Current git hash ({autohash}) does not match the one '
                        'used to generate the save directory ({old_git_hash}).'
                    )
        return current_hash, current_branch, autohash
    else:
        return None, None, None
