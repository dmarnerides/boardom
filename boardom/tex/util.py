import types
from inspect import signature
from ordered_set import OrderedSet
from pylatex.base_classes.command import Parameters
from pylatex.base_classes import (
    CommandBase,
    Command,
    UnsafeCommand,
    Options,
    Arguments,
    SpecialOptions,
    Container,
    Environment,
    ContainerCommand,
    Float,
    LatexObject,
)
from pylatex.base_classes.latex_object import _CreatePackages
from pylatex.labelref import RefLabelBase, Cref, CrefUp
from pylatex.lists import List
from pylatex.section import Part, Chapter, Paragraph, Subparagraph
from pylatex.tikz import TikZObject
from pylatex import (
    NewPage,
    LineBreak,
    NewLine,
    HFill,
    HugeText,
    LargeText,
    MediumText,
    SmallText,
    FootnoteText,
    TextColor,
    Document,
    Figure,
    SubFigure,
    StandAloneGraphic,
    MdFramed,
    FBox,
    Head,
    Foot,
    Marker,
    Label,
    Pageref,
    Ref,
    Eqref,
    Autoref,
    Hyperref,
    Enumerate,
    Itemize,
    Description,
    Alignat,
    Math,
    VectorName,
    Matrix,
    Package,
    HorizontalSpace,
    VerticalSpace,
    Center,
    FlushLeft,
    FlushRight,
    MiniPage,
    TextBlock,
    Quantity,
    Section,
    Subsection,
    Subsubsection,
    Tabular,
    Tabularx,
    MultiColumn,
    MultiRow,
    Table,
    Tabu,
    LongTable,
    LongTabu,
    LongTabularx,
    ColumnType,
    TikZOptions,
    TikZ,
    Axis,
    TikZScope,
    TikZCoordinate,
    TikZNodeAnchor,
    TikZNode,
    TikZUserPath,
    TikZPathList,
    TikZPath,
    TikZDraw,
    Plot,
)
from pylatex.utils import NoEscape, bold, italic, verbatim

__all__ = [
    'CommandBase',
    'Command',
    'UnsafeCommand',
    'Parameters',
    'Options',
    'Arguments',
    'SpecialOptions',
    'Container',
    'Environment',
    'ContainerCommand',
    'Float',
    'LatexObject',
    '_CreatePackages',
    'NewPage',
    'LineBreak',
    'NewLine',
    'HFill',
    'HugeText',
    'LargeText',
    'MediumText',
    'SmallText',
    'FootnoteText',
    'TextColor',
    'Document',
    'Figure',
    'SubFigure',
    'StandAloneGraphic',
    'MdFramed',
    'FBox',
    'Head',
    'Foot',
    'Marker',
    'RefLabelBase',
    'Label',
    'Ref',
    'Pageref',
    'Eqref',
    'Cref',
    'CrefUp',
    'Autoref',
    'Hyperref',
    'List',
    'Enumerate',
    'Itemize',
    'Description',
    'Alignat',
    'Math',
    'VectorName',
    'Matrix',
    'Package',
    'HorizontalSpace',
    'VerticalSpace',
    'Center',
    'FlushLeft',
    'FlushRight',
    'MiniPage',
    'TextBlock',
    'Quantity',
    'Section',
    'Part',
    'Chapter',
    'Subsection',
    'Subsubsection',
    'Paragraph',
    'Subparagraph',
    'Tabular',
    'Tabularx',
    'MultiColumn',
    'MultiRow',
    'Table',
    'Tabu',
    'LongTable',
    'LongTabu',
    'LongTabularx',
    'ColumnType',
    'TikZOptions',
    'TikZ',
    'Axis',
    'TikZScope',
    'TikZCoordinate',
    'TikZObject',
    'TikZNodeAnchor',
    'TikZNode',
    'TikZUserPath',
    'TikZPathList',
    'TikZPath',
    'TikZDraw',
    'Plot',
    'NoEscape',
    'bold',
    'italic',
    'verbatim',
    'DocumentBase',
    'NewCommand',
    'RenewCommand',
    'Block',
    'Huge',
    'TextSC',
    'HRef',
    'NLine',
    'Size',
    'HyperSetup',
    'IncludeGraphics',
    'DefineColor',
    'GraphicsPath',
    'Rule',
    'Centering',
    'Linewidth',
    'Dimexpr',
    'HSpace',
    'VSpace',
    'SetToWidth',
    'NewLength',
    'Emph',
    'PadMiniPage',
    'PushMiniPage',
    'Bold',
]


# These are to make containers a context manager
def cont_enter(self):
    return self


def cont_exit(self, *args, **kwargs):
    pass


def cont_add(self, *args):
    self += args
    return self


# Also .add function

Container.__enter__ = cont_enter
Container.__exit__ = cont_exit
Container.add = cont_add


# Addition of commands
def new_add(self, other):
    my_old_dumps = self.dumps
    if hasattr(other, 'dumps'):
        other_old_dumps = other.dumps
    elif isinstance(other, str):

        def other_old_dumps():
            return other

    else:
        raise TypeError

    def my_new_dumps(self):
        return my_old_dumps() + other_old_dumps()

    self.dumps = types.MethodType(my_new_dumps, self)
    return self


def new_radd(self, other):
    my_old_dumps = self.dumps
    if hasattr(other, 'dumps'):
        other_old_dumps = other.dumps
    elif isinstance(other, str):

        def other_old_dumps():
            return other

    else:
        raise TypeError

    def my_new_dumps(self):
        return other_old_dumps() + my_old_dumps()

    self.dumps = types.MethodType(my_new_dumps, self)
    return self


CommandBase.__add__ = new_add
CommandBase.__radd__ = new_radd


# Patch repr of LatexObject
def new_repr(self):
    return NoEscape(self.dumps())


LatexObject.__repr__ = new_repr

# This is to be able to register commands after the
# Child class is initialized and the required commands are added
#  class MetaRegister(_CreatePackages):
#      def __call__(cls, *args, **kwargs):
#          instance = super().__call__(*args, **kwargs)
#          for c in NewCommand.commands.values():
#              instance.preamble.append(c)
#          return instance


def Size(x, unit='pt'):
    if not x:
        return '1pt'
    if unit not in ['cm', 'pt', 'ex']:
        raise ValueError
    if isinstance(x, (int, float)):
        return NoEscape(f'{x} {unit}')
    elif isinstance(x, str):
        return NoEscape(x)
    elif hasattr(x, 'dumps'):
        return NoEscape(x.dumps())
    else:
        raise ValueError(f'Could not convert {str(x)} ({type(x)}) to size')


def simple_document(
    content, name='pdf_out', package_list=None, init_opts=None, generate_tex=False
):
    if package_list is None:
        package_list = []
    if init_opts is None:
        init_opts = dict(document_options=Size(12, 'pt'), lmodern=False)

    class _Doc(DocumentBase):
        packages = package_list

        def __init__(self):
            super().__init__(**init_opts)

        def create_body(self):
            return content

    doc = _Doc()
    doc.generate_pdf(filepath=name)
    if generate_tex:
        doc.generate_tex(filepath=name)
    return doc


class DocumentBase(Document):
    def __init__(self, *args, **kwargs):
        preamble = self.create_preamble()
        body = self.create_body()
        self.packages |= NewCommand.packages

        super(DocumentBase, self).__init__(*args, **kwargs)
        # 1. newcommands 2. preamble 3. Body

        for p in NewCommand.provide_commands:
            self.preamble.append(Command('providecommand', [NoEscape(rf'\{p}'), '']))
        for length in NewCommand.provide_lengths:
            self.preamble.append(NewLength(length))
        self.preamble += preamble
        self += body
        for c in NewCommand.commands.values():
            self.preamble.append(c)

    def create_preamble(self):
        return []
        #  pass
        #  raise NotImplementedError

    def create_body(self):
        return []
        #  pass
        #  raise NotImplementedError


class NewCommand(CommandBase):
    commands = {}
    renew = False
    sizes = {}
    data = {}
    provide_commands = OrderedSet()
    provide_lengths = OrderedSet()

    def __getattr__(self, key):
        if key in self.sizes:
            return Size(self.sizes[key])
        elif key in self.data:
            return self.data[key]

    def set_size(self, key, val):
        if key not in self.sizes:
            raise KeyError(f'No size called {key}')
        self.sizes[key] = Size(val)

    def __init__(self, *params, data=None):
        # Disable escaping for now
        old_escape = LatexObject.escape
        LatexObject.escape = False
        command_name = self.__class__.__name__.lower()
        args = list(signature(self.definition).parameters.keys())
        num_args = len(args)
        if len(params) != num_args:
            raise ValueError(
                f'Expected {num_args} parameters from '
                f'definition, but got {len(params)}.'
            )
        if data is not None:
            if not set(data.keys()).issubset(set(self.data.keys())):
                raise KeyError('Data contains invalid keys')
            self.data.update(data)
        if hasattr(self, 'packages'):
            NewCommand.packages |= self.packages
        if hasattr(self, 'provide_commands'):
            NewCommand.provide_commands |= self.provide_commands
        if hasattr(self, 'provide_lengths'):
            NewCommand.provide_lengths |= self.provide_lengths
        # Bind values from __init__ args to self before calling definition
        for name, p in zip(args, params):
            setattr(self, f'__{name}', p)
        definition = self.definition(*params)
        if isinstance(definition, (list, tuple)):
            definition = ''.join(
                [
                    NoEscape(x.dumps()) if hasattr(x, 'dumps') else str(x)
                    for x in definition
                ]
            )
        elif hasattr(definition, 'dumps'):
            definition = definition.dumps()
        definition = NoEscape(definition)
        if command_name not in NewCommand.commands:
            args_str = None if num_args == 0 else NoEscape(f'{num_args}')
            registry_command = Command(
                'renewcommand' if self.renew else 'newcommand',
                arguments=Command(command_name),
                options=args_str,
                extra_arguments=definition,
            )
            NewCommand.commands[command_name] = registry_command
        super(NewCommand, self).__init__(arguments=params)
        LatexObject.escape = old_escape


class RenewCommand(NewCommand):
    renew = True


class Block(Environment):
    omit_if_empty = True

    def __init__(self, *, data=None, **kwargs):
        super().__init__(data=data, **kwargs)

    def dumps(self):
        content = self.dumps_content()
        if not content.strip() and self.omit_if_empty:
            return ''
        cs = self.content_separator
        return '{' + cs + content + cs + '}'


class Huge(CommandBase):
    pass


class TextSC(CommandBase):
    packages = [Package('libertine')]


class HRef(CommandBase):
    packages = [Package('hyperref')]

    def __init__(self, link, content):
        super(HRef, self).__init__(arguments=[link, content])


class HyperSetup(CommandBase):
    packages = [Package('hyperref')]

    def __init__(self, *args):
        super(HyperSetup, self).__init__(arguments=','.join(args))


IncludeGraphics = StandAloneGraphic


class GraphicsPath(CommandBase):
    packages = [Package('graphicx')]

    def __init__(self, *args):
        path = NoEscape(Arguments(args).dumps())
        super(GraphicsPath, self).__init__(arguments=path)


class DefineColor(CommandBase):
    packages = [Package('xcolor')]

    def __init__(self, name, model, description):
        super(DefineColor, self).__init__(arguments=[name, model, description])


class Rule(CommandBase):
    def __init__(self, width, height, raise_size=None):
        super(Rule, self).__init__(arguments=[width, height], options=raise_size)


class Centering(CommandBase):
    pass


class Linewidth(CommandBase):
    pass


def Emph(CommandBase):
    pass


class Dimexpr(CommandBase):
    packages = [Package('calc')]

    def dumps(self):
        arguments = self.arguments._format_contents('(', ')(', ')')
        return rf"\{self.latex_name}{arguments}"


class SetToWidth(CommandBase):
    def __init__(self, name, text):
        super(SetToWidth, self).__init__(
            arguments=[NoEscape(rf'\{name}'), NoEscape(text)]
        )


class NewLength(CommandBase):
    def __init__(self, name):
        super(NewLength, self).__init__(NoEscape(rf'\{name}'))


HSpace = HorizontalSpace
VSpace = VerticalSpace


def NLine():
    return NoEscape(r'\\')


class PadMiniPage(NewCommand):
    def definition(self, space_before, space_after, content):
        before = HSpace('#1')
        after = HSpace('#2')
        mpage = MiniPage(width=Dimexpr(r'\linewidth - #1 - #2')).add(r'#3')
        return MiniPage(width=Linewidth()).add(before, mpage, after)


class PushMiniPage(NewCommand):
    def definition(self, space_before, size, content):
        before = HSpace('#1')
        mpage = MiniPage(width='#2').add(r'#3')
        return MiniPage(width=Dimexpr('#1 + #2')).add(before, mpage)


class Bold(NewCommand):
    def definition(self, content):
        return r'\textbf{#1}'
