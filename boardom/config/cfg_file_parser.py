import os
import boardom as bd
from .common import (
    Group,
    UNTOUCHABLES,
    _is_valid_argname,
    AUTOMATIC_ARGS,
)

# NOTE: THESE COMMENTS ARE OUTDATED
# !!! The logic relies on the dicts to be ordered in this file.

# ######################################################################
# This allows comments, tagging and categories in the configuration files
# Comments start with #
# configuration elements start with -- and occupy only one line
# in the configuration file
# They can also not have the -- tags
# e.g.
# --lr 0.1
# OR
# lr 0.1
# tags are enclosed with <>, i.e. <tag>
# tags can be at the end of an argument line
# e.g.
# --lr 0.01 <hyper>
# --batch_size 128 <hyper>
# tag blocks can be created like so
# <hyper>
# --lr 0.01
# --batch_size 128
# <>
# category marking works like tags
# e.g.
# --lr 0.01 {generator}
# --batch_size 0.01 {generator}
# OR
# {generator}
# --lr 0.01
# --beta1 0.9
# {}
# multiple tags/categories must be within the
# same brackets separated by commas ,
# e.g. {generator, discriminator} or <hyper, super>
# or
# {generator}
# {discriminator}
# --lr 0.01
# {}
# --beta1 0.9
# {}
# but not {generator} {discriminator} or <hyper,super>
# Open category sections and tags in different lines
#
# #################################################################
# If a parameter was at any point added into a category then all
# lookups of that parameter in the program will need a category
# supplied. The default is suppressed
# #################################################################

_CONFIG_EXTENSIONS = ['.bdc', '.bd']


def _is_valid_config_file(x):
    ret = any([x.lower().endswith(y) for y in _CONFIG_EXTENSIONS])
    return os.path.isfile(bd.process_path(x)) and ret


class ConfigFileParser:
    def __init__(self):
        self.elements_store = {
            # The list element stores the current active tags/Group as the configs are being parsed
            'tag': ('<', '>', []),
            'group': ('{', '}', []),
        }
        self.all_groups = set()

    def parse_cfg_file(self, config_file):
        yield from self._parse_cfg_file(config_file, subfile=False)

    def _parse_cfg_file(self, config_file, subfile=False):
        config_file = bd.process_path(config_file)
        if not os.path.isfile(config_file):
            raise FileNotFoundError(config_file)
        with open(config_file) as f:
            for count, line in enumerate(f):
                yield from self._process_line(line.strip('\n'), count + 1, config_file)
        # Avoid checking opened if it's a subfile
        if subfile:
            return
        for element_type, (_, _, current_list) in self.elements_store.items():
            if current_list:
                elts = ', '.join([f'\'{x}\'' for elm in current_list for x in elm])
                raise SyntaxError(
                    f'Following {element_type}(s) were not closed\n\t{elts}'
                    f'\nin file: {config_file}'
                )

    def _process_line(self, line, count, config_file):
        basename = os.path.basename(config_file)
        directory = os.path.dirname(config_file)
        # Clean line and get only relevant bits
        clean_line = self._clean_line(line, count, basename)
        if not clean_line:
            return
        # If we reach here then it must be an argument line
        extras, clean_line = self._get_line_extras(clean_line, count, basename)
        # clean line should now be either a (relative) config file
        # or a name - value pair
        arg_name, is_filename = self._get_arg_name(clean_line, line, count, basename)
        if is_filename:
            # config files should be relative to another
            fname = os.path.join(directory, arg_name)
            fname = os.path.normpath(fname)
            yield from self._parse_cfg_file(fname, subfile=True)
            return
        value = self._get_arg_value(clean_line)

        # Get all tags/groups
        active_tags = self._get_active_tags(extras, arg_name)
        active_groups = self._get_active_groups(extras, arg_name)
        self.all_groups = self.all_groups.union(active_groups)

        # Create the current data dictionary
        yield value, arg_name, active_groups, active_tags, line, count, config_file

    def _line_error(self, line, count, basename):
        raise SyntaxError(
            f'Could not parse line {count} in file ' f'\'{basename}\':\n\t\'{line}\''
        )

    def _check_valid_line(self, line, count, basename):
        for word, (starting, ending, _) in self.elements_store.items():
            s_pos = line.find(starting)
            if s_pos >= 0:
                if line[s_pos + 1 :].find(starting) >= 0:
                    raise SyntaxError(
                        f'{word.capitalize()} opened twice in file '
                        f'\'{basename}\', line {count}:\n\t{line}'
                    )
                e_pos = line.find(ending)
                if e_pos < 0:
                    raise SyntaxError(
                        f'{word.capitalize()} opened but not closed '
                        f'in \'{basename}\', line {count}:\n\t{line}'
                        f'\nExpected {ending}.'
                    )
                if line[e_pos + 1 :].find(ending) >= 0:
                    raise SyntaxError(
                        f'{word.capitalize()} closed twice in '
                        f'\'{basename}\', line {count}:\n\t{line}'
                    )

    def _get_annotation_elements(self, line, element_type, count, basename):
        s_open, s_close, _ = self.elements_store[element_type]
        if line.find(s_open) >= 0:
            elements = line[line.find(s_open) + 1 : line.find(s_close)]
            elements = [x.strip() for x in elements.split(',')]
            elements = [x for x in elements if x != '']
            for x in elements:
                if not x.isidentifier():
                    raise SyntaxError(
                        f'Invalid {element_type} \'{x}\' in file '
                        f'\'{basename}\', line {count}.'
                        f'\n\t\'{line}\'\nMust be a valid Python identifier.'
                    )
            # Clean the line
            line = line[: line.find(s_open)] + line[line.find(s_close) + 1 :]
        else:
            elements = None
        return elements, line

    # This adjusts the section opening / closing and also registers
    def _is_section_open_or_close(self, line, count, basename):
        opening_chars = [x[0] for x in self.elements_store.values()]
        line_list = line.split()
        clean_line = line
        if any([line_list[0][0] == x for x in opening_chars]):
            for element_type, (_, _, current_list) in self.elements_store.items():
                new_elements, clean_line = self._get_annotation_elements(
                    clean_line, element_type, count, basename
                )
                if new_elements is None:
                    continue
                elif not new_elements:
                    if len(current_list) == 0:
                        raise SyntaxError(
                            f'Attempting to close {element_type} '
                            f'before opening '
                            f'in file \'{basename}\','
                            f' line {count}.'
                        )
                    del current_list[-1]
                else:
                    current_list.append(new_elements)
            if clean_line.split('#')[0].strip() != '':
                self._line_error(line, count, basename)
            return True
        else:
            return False

    def _clean_line(self, line, count, basename):
        line = line.strip()
        line_list = line.split()
        # Empty line or comments line
        if not line_list or line_list[0][0] == '#':
            return ''
        self._check_valid_line(line, count, basename)
        # Group section opening / closing
        if self._is_section_open_or_close(line, count, basename):
            return ''

        # Remove any comments
        return line.split('#')[0]

    def _get_line_extras(self, clean_line, count, basename):
        extras = {}
        for element_type in self.elements_store:
            current_elements, clean_line = self._get_annotation_elements(
                clean_line, element_type, count, basename
            )
            extras[element_type] = current_elements or []
        return extras, clean_line

    def _get_arg_name(self, clean_line, line, count, basename):
        arg_list = clean_line.split()
        if len(arg_list) == 1:
            return arg_list[0], True
        arg_name = arg_list[0]
        # Allow '--argument'
        if arg_name.startswith('--'):
            arg_name = arg_name[2:]
        # also allow 'argument:'
        elif arg_name.endswith(':'):
            arg_name = arg_name[:-1]
        if not _is_valid_argname(arg_name):
            raise SyntaxError(
                f'Invalid argument name \'{arg_name}\' in file '
                f'\'{basename}\', line {count}:\n\t\'{line}\''
            )
        if arg_name in AUTOMATIC_ARGS:
            raise SyntaxError(
                f'Can not set \'{arg_name}\' from configuration files. '
                f'File \'{basename}\', line {count}:\n\t\'{line}\''
            )

        return arg_name, False

    def _get_arg_value(self, clean_line):
        arg_list = clean_line.split()
        value = ' '.join(arg_list[1:])
        # strip any parentheses, square brackets and split at commas
        # (to support arrays and tuples)
        for c in '[],()':
            value = value.replace(c, ' ')
        # Make it compact:
        value = ' '.join(value.split())
        return value

    def _get_active_tags(self, extras, arg_name):
        tag_list = self.elements_store['tag'][2]
        tags = [item for sublist in tag_list for item in sublist]
        tags = set(tags + extras['tag'])
        if tags and (arg_name in UNTOUCHABLES):
            raise SyntaxError(f'{arg_name} can not have a tag.')
        return tags

    def _get_active_groups(self, extras, arg_name):
        group_list = self.elements_store['group'][2]
        groups = [item for sublist in group_list for item in sublist]
        groups = Group(groups + extras['group'])
        if groups and (arg_name in UNTOUCHABLES):
            raise SyntaxError(f'{arg_name} can not be in a custom group.')
        return groups

    def _warn_override(self, active_groups, arg_name, basename, count, line):
        line = line.split('#')[0].strip()
        if active_groups.is_default:
            group_warn = 'in default group'
        else:
            group_warn = 'in groups: (' + ', '.join(active_groups) + ')'
        line_info = f'File \'{basename}\', line {count}: \'{line}\'.'
        over = 'Overwriting...'
        bd.warn(f'{arg_name} defined twice {group_warn}. {line_info} {over}')
