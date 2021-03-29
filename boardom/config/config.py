import time
import argparse
import sys
import itertools
from copy import deepcopy
from contextlib import contextmanager
import boardom as bd
from ..multiprocessing import _PROCESS_ID
from .common import (
    Group,
    UNTOUCHABLES,
    DEFAULT_CFG_DICT,
    CORE_ARGNAMES,
    AUTOMATIC_ARGS,
    _is_valid_argname,
    _create_datum,
)
from .cfg_file_parser import ConfigFileParser, _is_valid_config_file
from .session_helpers import _create_session

# TODO: Allow store_true type arguments

# (To consider): Groups are only defined in config files.
#       This can be an issue if there is a typo in the config files.
#       Especially in this case:
#       --lr 0.1 {generator}
#       --foo 0.2 {generatoe}
#       Alleviated by printing all the groups when running setup
#       Also alleviated by the default no fallback mechanism,
#       so the value 0.2 is epected from cfg.g.generator.foo


_CONFIG_OBJ_NAMES = set(AUTOMATIC_ARGS + ['_prv'])


class ConfigGetError(RuntimeError):
    pass


class ConfigSetError(RuntimeError):
    pass


_GCTX_RESERVED = set(['_GCTX_CFG_OBJ', '_GCTX_IS_DEFAULT', '_GCTX_OLD_IDS'])


class GroupContext:
    def __init__(self, cfg_obj, group=None, old_ctx=None, default=False):
        self._GCTX_IS_DEFAULT = default
        self._GCTX_CFG_OBJ = cfg_obj
        if not default:
            if group is None:
                new_group = []
            else:
                new_group = [group]

            old_groups = cfg_obj._prv['group_ctx_dict'].get(id(old_ctx), [])
            new_group = old_groups + new_group
            # Sort and get unique (set() does not preserve order)
            new_group = list(dict.fromkeys(new_group))
            cfg_obj._prv['group_ctx_dict'][id(self)] = new_group
        else:
            self._GCTX_OLD_IDS = None

    def __del__(self):
        if not self._GCTX_IS_DEFAULT:
            del self._GCTX_CFG_OBJ._prv['group_ctx_dict'][id(self)]

    def __enter__(self):
        if not self._GCTX_IS_DEFAULT:
            self._GCTX_CFG_OBJ._prv['current_group_ctx_ids'].append(id(self))
        else:
            self._GCTX_OLD_IDS = self._GCTX_CFG_OBJ._prv['current_group_ctx_ids']
            self._GCTX_CFG_OBJ._prv['current_group_ctx_ids'] = []

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._GCTX_IS_DEFAULT:
            self._GCTX_CFG_OBJ._prv['current_group_ctx_ids'].pop()
        else:
            self._GCTX_CFG_OBJ._prv['current_group_ctx_ids'] = self._GCTX_OLD_IDS

    def set(self, arg_name, value, force=False):
        with self:
            return self._GCTX_CFG_OBJ.set(arg_name, value, force=force)

    def get(self, arg_name, default=bd.Null):
        with self:
            return self._GCTX_CFG_OBJ.get(arg_name, default=default)

    def __getattr__(self, key):
        with self:
            return getattr(self._GCTX_CFG_OBJ, key)

    def __setattr__(self, key, val):
        if key in _GCTX_RESERVED:
            super().__setattr__(key, val)
        else:
            with self:
                setattr(self._GCTX_CFG_OBJ, key, val)

    def __getitem__(self, key):
        with self:
            return self._GCTX_CFG_OBJ[key]

    def __setitem__(self, key, val):
        with self:
            self._GCTX_CFG_OBJ[key] = val

    @property
    def g(self):
        return GroupContextGenerator(self._GCTX_CFG_OBJ, self)

    def __str__(self):
        return f'GroupContext({", ".join(self._GCTX_GROUPS)})'


class GroupContextGenerator:
    def __init__(self, cfg_obj, group_ctx=None):
        self._GCTX_CFG_OBJ = cfg_obj
        self._GCTX = group_ctx

    def __getattr__(self, group):
        return GroupContext(self._GCTX_CFG_OBJ, group, self._GCTX)

    def __setattr__(self, key, val):
        if key in ['_GCTX_CFG_OBJ', '_GCTX']:
            super().__setattr__(key, val)
        else:
            raise RuntimeError('Can not set attribute of GroupContextGenerator object')

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(str(e)) from None

    def __setitem__(self, key):
        raise RuntimeError('Can not set item of GroupContextGenerator object')


class Config:
    def __init__(self):
        self._prv = dict(
            # The data needed to generate the argparse parser
            # {argname: {grp: {value, tags etc}}}
            data={},
            # The list of all .bd files used (in order)
            cfg_files=[],
            # The base arguments used to build the parser
            # e.g. dict(flag='--lr', default=1e-3)
            argparse_arguments=[],
            # The set of argument lists to use from the defaults
            argument_prebaked_categories=set(),
            # Flag to determine if setup is done
            done_setup=False,
            # Set of all groups
            all_groups=set(),
            # Default is to not fall back groups
            group_fallback=False,
            # Group stack used when subgrouping etc
            group_ctx_dict={},
            current_group_ctx_ids=list(),
            #  # Whether this is the global bd.cfg
            #  is_core_config=len(bd._CFG_OBJ_LIST) == 0,
            # Whether it actually has the core configuration arguments,
            has_core_config=False,
        )

        # Initialize with empty argument list
        self.empty_argument_list()
        bd._CFG_OBJ_LIST.append(self)

    def _use_argument_categories(self, *arg_cats, empty=True):
        if self._prv['done_setup']:
            raise RuntimeError('Attempted to change configuration after setup.')
        if empty:
            self._prv['argparse_arguments'] = []
            self._prv['argument_prebaked_categories'] = set()
        if 'core' in arg_cats:
            self._prv['has_core_config'] = True
        existing_flags = [x['flag'][2:] for x in self._prv['argparse_arguments']]
        for arg in arg_cats:
            if arg in DEFAULT_CFG_DICT:
                new_args = deepcopy(DEFAULT_CFG_DICT[arg])
                new_arg_flags = set(x['flag'][2:] for x in new_args)
                for ef in existing_flags:
                    if ef in new_arg_flags:
                        bd.warn(f'Overriding previously defined argument: {ef}')
                self._prv['argparse_arguments'] += new_args
                self._prv['argument_prebaked_categories'].add(arg)
            else:
                raise ValueError(f'Unknown configuration: {arg}')

    def empty_argument_list(self):
        self._use_argument_categories()

    def use_boardom_arguments(self, *arglist_names):
        # If arglist names is empty, use all categories
        args = arglist_names
        #  is_core_config = self._prv['is_core_config']
        if not args:
            args = list(DEFAULT_CFG_DICT.keys())
        args = list(args)
        #  if (not is_core_config) and ('core' in args):
        #      args.remove('core')

        self._use_argument_categories(*args, empty=False)

    def add_arguments(self, *args, override=False):
        if self._prv['done_setup']:
            raise RuntimeError('Attempted to add argument after setup.')
        current_flags = {
            x['flag'][2:]: i for i, x in enumerate(self._prv['argparse_arguments'])
        }
        for arg in args:
            arg_name = arg['flag']
            if arg_name.startswith('--'):
                arg_name = arg_name[2:]
            else:
                # Add the dashes if missing
                arg['flag'] = f'--{arg_name}'
            if arg_name.startswith('-') or (not _is_valid_argname(arg_name)):
                raise RuntimeError(f'Argument {arg_name} is invalid.')

            if arg_name in CORE_ARGNAMES + AUTOMATIC_ARGS:
                msg = f'Argument \'{arg_name}\' is in the core arguments.'
                if override:
                    msg = msg[:-1] + ' and can not be overriden.'
                raise RuntimeError(msg)

            # Don't allow setting things in dir(self)
            if arg_name in dir(self):
                raise RuntimeError('Can not set {} as an argument.')

            if arg_name in current_flags:
                if override:
                    bd.warn(f'Overriding {arg_name} for argparse')
                    del self._prv['argparse_arguments'][current_flags[arg_name]]
                else:
                    raise RuntimeError(
                        f'Argument \'{arg_name}\' already defined. '
                        f'Existing flags can be overridden by '
                        f'passing override=True to add_args'
                    )
            self._prv['argparse_arguments'].append(arg)

    def add_argument(self, override=False, **kwargs):
        self.add_arguments(kwargs, override=override)

    def _parse(self, arglist):
        # First parse all the input files and also get extra arguments
        # The data from the config files goes into self._prv['data']
        cfg_file_parser = ConfigFileParser()
        extra_argv = []
        for x in arglist:
            if _is_valid_config_file(x):
                fname = bd.process_path(x)
                self._prv['cfg_files'].append(fname)
                for data in cfg_file_parser.parse_cfg_file(fname):
                    self._add_cfg_file_line_data(data)
            else:
                extra_argv.append(x)
        self._prv['all_groups'] = cfg_file_parser.all_groups

        # Check that all arguments provided in the config files are registered for the parser
        # If not, give meaningful errors.
        # This is automatically handled by the parser, but we do this here to easily track
        # the config file error
        all_argnames = {
            x['flag'][2:]: i for i, x in enumerate(self._prv['argparse_arguments'])
        }
        for arg_name in self._prv['data']:
            if arg_name not in all_argnames:
                for arg_data in self._prv['data'][arg_name].values():
                    meta = arg_data['meta']
                    file, line, count = meta['file'], meta['line'], meta['count']
                    raise RuntimeError(
                        f'Could not find registered argument for \'{arg_name}\' '
                        f'provided in \'{file}\', line {count}:\n\t{line}'
                    )

        # Create parser and all arguments
        parser = argparse.ArgumentParser(allow_abbrev=False, conflict_handler='resolve')
        for arg in self._prv['argparse_arguments']:
            base_flag_name = arg['flag']
            arg_name = arg['flag'][2:]
            help_str = arg['help'] if 'help' in arg else None
            if arg_name.startswith('_'):
                raise ValueError(f'Argument \'{arg_name}\' starts with "_".')

            parser.add_argument(
                base_flag_name,
                help=help_str,
                **{k: v for k, v in arg.items() if k not in ['flag', 'help']},
            )

            # Also add all grouped versions of the argument, defined in the config files
            if arg_name in self._prv['data']:
                for group in self._prv['data'][arg_name]:
                    # Ignore default group (already added)
                    if group.is_default:
                        continue
                    flag_name = group.build_full_argname(base_flag_name)
                    if help_str and not group.is_default:
                        help_str += f' ({group})'
                    kwargs = {k: v for k, v in arg.items() if k not in ['flag', 'help']}
                    parser.add_argument(flag_name, help=help_str, **kwargs)

        # Generate argv for parser
        arg_list = []
        for arg_name, data in self._prv['data'].items():
            for group, values in data.items():
                name = group.build_full_argname(arg_name)
                for x in [f'--{name}'] + values['value']:
                    arg_list.append(x)

        return parser.parse_args(arg_list + extra_argv)

    def _add_cfg_file_line_data(self, data):
        value, arg_name, group, tags, line, count, config_file = data

        arg_dict = self._prv['data'].get(arg_name, {})
        group_dict = arg_dict.get(group, {})

        group_dict['value'] = value.split()
        group_dict['tags'] = tags
        group_dict['meta'] = {'file': config_file, 'line': line, 'count': count}

        arg_dict[group] = group_dict
        self._prv['data'][arg_name] = arg_dict

    def _update_data_from_parsed(self, parsed_data):
        data = self._prv['data']
        for key, val in vars(parsed_data).items():
            arg_name, group = Group.from_full_argname(key)
            if (arg_name in data) and (group in data[arg_name]):
                data[arg_name][group]['value'] = val
            else:
                datum = _create_datum(val)
                if arg_name not in data:
                    data[arg_name] = datum
                else:
                    data[arg_name][Group()] = datum[Group()]

    # Core config uses sysargv by default,
    # Others are False by default
    # extra can be None or list
    def setup(self, *cfg_files, extra=None, use_sysargv=True):
        if not self._prv['done_setup']:
            bd.log('Processing configuration')
            arglist = []
            #  if use_sysargv is bd.Null:
            #      use_sysargv = self._prv['is_core_config']
            if not isinstance(use_sysargv, bool):
                raise RuntimeError(
                    f'use_sysargv expected a bool value. Got {type(use_sysargv)}'
                )
            if use_sysargv:
                arglist += sys.argv[1:]
            if cfg_files:
                cfg_files = [bd.process_path(f) for f in cfg_files]
                arglist += cfg_files
            if extra is not None:
                arglist += extra

            self._update_data_from_parsed(self._parse(arglist))

            all_groups = self._prv['all_groups']
            if all_groups:
                bd.log(f'Groups defined: {self._prv["all_groups"]}')

            # Register automatic arguments
            self._prv['data']['time_configured'] = _create_datum(
                time.strftime("%Y/%m/%d %H:%M:%S")
            )
            self._prv['data']['process_id'] = _create_datum(_PROCESS_ID)
            self._prv['data']['session_path'] = _create_datum(None)

            # Leave this here, as the Logger functions called later on (in the subprocess)
            # and accesing cfg.project_path and cfg.session_name
            # depend on correctly identifying if _prv['done_setup'] is True or False
            self._prv['done_setup'] = True

            #  # If using logger, notify with session_id.  This is to change the ID
            #  # from the execution_id to the session_path
            #  if bd.BoardomLogger._started:
            #      # CFG needs to be sent first (lmdb requires session_path)
            #      bd.BoardomLogger()._send_cfg_full()
            #      bd.BoardomLogger()._start_lmdb()
            bd.log('Config done.')

        elif cfg_files:
            raise RuntimeError(
                'Could not setup from config files as bd.setup() was already called.'
            )

        return self

    def create_session(self, session_name=None):
        return _create_session(self, session_name=session_name)

    # This is the latest group list in the stack
    @property
    def current_group(self):
        self.setup()
        return Group(self._get_current_group_ctx())

    def _get_current_group_ctx(self):
        ctx_ids = self._prv['current_group_ctx_ids']
        ctx_dict = self._prv['group_ctx_dict']
        ret = [x for ctx_id in ctx_ids for x in ctx_dict[ctx_id]]
        # Make entries unique but keeping order
        ret = list(dict.fromkeys(ret))
        return ret

    def _groups_generator(self):
        group_list = self._get_current_group_ctx()
        if group_list and self._prv['group_fallback']:
            # Get unique and sort descending for priority
            flat_stack = list(dict.fromkeys(group_list))[::-1]
            # TODO: OPTIMIZE COMBINATIONS WITH PRIORITY
            all_combs = [
                comb
                for comb_size in range(len(flat_stack), 0, -1)
                for comb in itertools.combinations(flat_stack, comb_size)
            ]
            sorted_combs = [
                comb for word in flat_stack for comb in all_combs if comb[0] == word
            ]
            for comb in sorted_combs:
                yield Group(comb)
            # Also yield default when doing fallback
            group_list = []
        yield Group(group_list)

    # Context manager for group fallback
    @contextmanager
    def group_fallback(self, fallback=True):
        self.setup()
        if not isinstance(fallback, bool):
            raise ValueError('Expected bool value for fallback')
        old_fallback = self._prv['group_fallback']
        self._prv['group_fallback'] = fallback
        try:
            yield
        finally:
            self._prv['group_fallback'] = old_fallback

    # This gives a group context (generator) with nesting
    # i.e. grp1 = cfg.g.grp1
    @property
    def g(self):
        self.setup()
        return GroupContextGenerator(self)

    # This gives the default group context
    # i.e. def_grp = cfg.dg
    @property
    def dg(self):
        self.setup()
        return GroupContext(self, default=True)

    def get(self, arg_name, default=bd.Null):
        self.setup()
        arg_data = self._prv['data'].get(arg_name, {})
        count = 0
        for group in self._groups_generator():
            count += 1
            if group in arg_data:
                return arg_data[group]['value']
        else:
            # If default is provided it should not be Null
            if default is not bd.Null:
                return default
            # else throw errors
            is_default_group = (count == 1) and (group.is_default)
            extra_info = ''
            if not is_default_group:
                extra_info += f' (groups: {", ".join(self._get_current_group_ctx())})'
            raise ConfigGetError(f'Argument not defined: {arg_name}{extra_info}')

    def set(self, arg_name, value, force=False):
        if (arg_name in UNTOUCHABLES) and (not force):
            raise ConfigSetError(f'Argument can not be changed: {arg_name}')
        self.setup()
        group = self.current_group
        data = self._prv['data']
        if arg_name not in data:
            raise ConfigSetError(f'Invalid argument name: {arg_name}')

        arg_dict = data[arg_name]

        if group not in arg_dict:
            raise ConfigSetError(
                f'Group {group} was not defined for {arg_name} argument.'
            )

        arg_dict[group]['value'] = value
        #  if self._prv['is_core_config'] and bd.BoardomLogger._started:
        #      bd.BoardomLogger()._update_cfg_value(arg_name, str(group), value)

    def __getitem__(self, name):
        try:
            return self.get(name)
        except ConfigGetError as e:
            raise KeyError(str(e)) from None

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(str(e)) from None

    def __setitem__(self, name, value):
        # We first try to set from super() since .get is a bit expensive
        if name in _CONFIG_OBJ_NAMES:
            super().__setattr__(name, value)
            return
        try:
            return self.set(name, value)
        except ConfigSetError as e:
            raise KeyError(str(e)) from None

    def __setattr__(self, name, value):
        try:
            self[name] = value
        except KeyError as e:
            raise AttributeError(str(e)) from None

    def __str__(self):
        to_print = []
        for arg, arg_dict in self._prv['data'].items():
            for group, data in arg_dict.items():
                grp_str = ''
                if group:
                    grp_str = f' {{{group}}}'

                to_print.append(f'  {arg}: {data["value"]}{grp_str}')
        to_print.sort()
        content = '\n'.join(to_print)
        return f'Config(\n{content}\n)'

    def __len__(self):
        if self._prv['done_setup']:
            return sum(len(x) for x in self._prv['data'].values())
        else:
            return 0

    def __contains__(self, arg_name):
        if not self._prv['done_setup']:
            raise RuntimeError(
                'Attempted to check if config object contatins argument before setup().'
            )
        return arg_name in self._prv['data']

    def _get_data_dict(self):
        return {
            arg_name: {str(group): group_dict for group, group_dict in arg_dict.items()}
            for arg_name, arg_dict in self._prv['data'].items()
        }

    #  def _update(self, other_cfg):
    #      self.__dict__.update(other_cfg.__dict__)
