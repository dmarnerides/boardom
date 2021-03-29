import os
import shutil
import pytest
import sys
from collections.abc import Callable
from importlib import reload
import boardom as bd

_FILE = os.path.abspath(__file__)

_DIR = os.path.dirname(_FILE)

_ASSETS_DIR = os.path.join(_DIR, 'tmp_assets')

# TODO: Fallback logic testing
# TODO: bd File parsing testing


def setup_module():
    if os.path.exists(_ASSETS_DIR):
        shutil.rmtree(_ASSETS_DIR)
    os.mkdir(_ASSETS_DIR)


def teardown_module():
    if os.path.exists(_ASSETS_DIR):
        shutil.rmtree(_ASSETS_DIR)


def create_asset(content, ext):
    create_asset.counter += 1
    asset_fullname = os.path.join(_ASSETS_DIR, f'tmp_{create_asset.counter}{ext}')
    with open(asset_fullname, 'w') as f:
        f.write(content)
    return asset_fullname


create_asset.counter = 0


class TestConfig:
    @pytest.fixture(autouse=True)
    def refresh_bd_and_argv(self):
        reload(bd)
        sys.argv = [_FILE]
        for x in ['utils', 'utils.config']:
            if x in sys.modules:
                reload(x)

    def test_bd_cfg_exists_on_import(self):
        assert isinstance(bd.cfg, bd.config.config.Config)

    def test_bd_setup_exists_on_import(self):
        assert isinstance(bd.setup, Callable)

    def test_bd_setup_is_not_done_on_import(self):
        assert bd.cfg._prv['done_setup'] is False

    def test_can_call_setup_and_setup_returns_self(self):
        assert bd.cfg._prv['done_setup'] is False
        cfg = bd.setup()
        assert bd.cfg._prv['done_setup'] is True
        assert cfg is bd.cfg

    def test_calling_setup_with_wrong_sysargv_raises_error(self):
        sys.argv = sys.argv + ['bonkers']
        with pytest.raises(SystemExit):
            bd.setup()

    def test_subsequent_calls_to_setup_do_nothing(self):
        assert bd.cfg._prv['done_setup'] is False
        cfg = bd.setup()
        assert bd.cfg._prv['done_setup'] is True
        cfg = bd.setup()
        assert bd.cfg._prv['done_setup'] is True
        assert cfg is bd.cfg

    def test_can_create_other_config_objects(self):
        cfg = bd.setup()
        other = bd.config.config.Config()
        assert cfg is not other

    def test_can_set_global_config_module(self):
        from util.config import my_bd_cfg_is_same_as_given

        cfg = bd.setup()
        other = bd.config.config.Config()
        assert my_bd_cfg_is_same_as_given(cfg)
        assert not my_bd_cfg_is_same_as_given(other)
        bd.cfg = other
        assert my_bd_cfg_is_same_as_given(other)

    def test_can_not_set_global_config_module_if_imported_as_global(self):
        from util.imported_cfg import my_bd_cfg_is_same_as_given

        cfg = bd.setup()
        other = bd.config.config.Config()
        assert my_bd_cfg_is_same_as_given(cfg)
        assert not my_bd_cfg_is_same_as_given(other)
        bd.cfg = other
        assert not my_bd_cfg_is_same_as_given(other)

    def test_only_uses_automatic_arguments_by_default_when_imported(self):
        from boardom.config.common import AUTOMATIC_ARGS

        assert len(bd.cfg) == 0
        bd.setup()
        assert len(bd.cfg) == len(AUTOMATIC_ARGS)

    def test_can_add_argument_and_retrieve_it(self):
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        # Second form
        bd.cfg.add_arguments(dict(flag='--bar', default=1, help='Bar'))
        assert bd.cfg.foo == 42
        assert bd.cfg.bar == 1

    def test_can_add_argument_without_flag_dashes(self):
        bd.cfg.add_argument(flag='foo', default=42, help='Foo')
        assert bd.cfg.foo == 42

    def test_can_not_add_invalid_arguments(self):
        with pytest.raises(RuntimeError):
            bd.cfg.add_argument(flag='2foo', default=42, help='Foo')
        with pytest.raises(RuntimeError):
            bd.cfg.add_argument(flag='foo-', default=42, help='Foo')
        with pytest.raises(RuntimeError):
            bd.cfg.add_argument(flag='+3', default=42, help='Foo')
        # Core arguments can not be overriden
        from boardom.config.common import CORE_SETTINGS

        for x in CORE_SETTINGS:
            with pytest.raises(RuntimeError):
                bd.cfg.add_argument(**x)

    def test_adding_arguments_does_not_trigger_setup(self):
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        assert bd.cfg._prv['done_setup'] is False

    def test_first_read_of_argument_triggers_setup(self):
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        assert bd.cfg._prv['done_setup'] is False
        assert bd.cfg.foo == 42
        assert bd.cfg._prv['done_setup'] is True

    def test_can_not_add_arguments_after_setup(self):
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        bd.setup()
        with pytest.raises(RuntimeError) as e:
            bd.cfg.add_argument(flag='--bar', default=1, help='Bar')
        assert 'Attempted to add argument after setup' in str(e)

    def test_repeated_arguments_raise_error_unless_overriden(self):
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        with pytest.raises(RuntimeError) as e:
            bd.cfg.add_argument(flag='--foo', default=3, help='Foo')
        assert "Argument 'foo' already defined" in str(e)
        bd.cfg.add_argument(flag='--foo', default=3, help='Foo', override=True)
        assert bd.cfg.foo == 3

    def test_can_not_add_reserved_arguments(self):
        from boardom.config.common import CORE_ARGNAMES, AUTOMATIC_ARGS

        reserved = CORE_ARGNAMES + AUTOMATIC_ARGS
        for arg in reserved:
            with pytest.raises(RuntimeError) as e:
                bd.cfg.add_argument(flag=arg)
            assert f"Argument '{arg}' is in the core arguments" in str(e)

    def test_can_not_add_reserved_arguments_even_with_override(self):
        from boardom.config.common import CORE_ARGNAMES, AUTOMATIC_ARGS

        reserved = CORE_ARGNAMES + AUTOMATIC_ARGS
        for arg in reserved:
            with pytest.raises(RuntimeError) as e:
                bd.cfg.add_argument(flag=arg, override=True)
            assert f"Argument '{arg}' is in the core arguments" in str(e)

    def test_can_pass_argument_value_via_sysargv(self):
        sys.argv = sys.argv + ['--foo', '3']
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        assert bd.cfg.foo == '3'

    def test_type_restrictions_from_argparse_work(self):
        sys.argv = sys.argv + ['--foo', '3']
        bd.cfg.add_argument(flag='--foo', type=int, default=42, help='Foo')
        assert bd.cfg.foo == 3

    def test_can_call_setup_without_sysargv(self):
        sys.argv = sys.argv + ['bonkers']
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        bd.setup(use_sysargv=False)

    def test_calling_without_sysargv_ignores_sysargv_values(self):
        sys.argv = sys.argv + ['--foo', '3']
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        bd.setup(use_sysargv=False)
        assert bd.cfg.foo == 42

    def test_can_pass_list_to_setup_for_replacing_sysargv(self):
        sys.argv = sys.argv + ['--foo', '3']
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        bd.setup(use_sysargv=False, extra=['--foo', '5'])
        assert bd.cfg.foo == '5'

    def test_can_select_boardom_provided_arguments(self):
        sys.argv = sys.argv + ['--lr', '100000']
        bd.cfg.use_boardom_arguments()
        assert bd.cfg.lr == 100000

    def test_can_select_specific_boardom_provided_arguments(self):
        sys.argv = sys.argv + ['--lr', '3', '--lr_schedule', 'step']
        bd.cfg.use_boardom_arguments('extra')
        assert bd.cfg.lr == 3
        assert bd.cfg.lr_schedule == 'step'
        with pytest.raises(AttributeError):
            bd.cfg.session_name

    def test_can_check_if_config_contains_argument(self):
        sys.argv = sys.argv + ['--lr', '100000']
        bd.cfg.use_boardom_arguments()
        # If not done setup it raises RuntimeError:
        with pytest.raises(RuntimeError) as e:
            assert 'lr' not in bd.cfg
        assert 'Attempted to check' in str(e)

        bd.cfg.setup()
        assert 'lr' in bd.cfg
        assert 'non_existing_arg' not in bd.cfg
        assert bd.cfg.lr == 100000

    #  def test_can_only_create_one_core_config_and_bd_config_is_the_one(self):
    #      sys.argv = sys.argv + ['--session_name', 'foo_session']
    #      bd.cfg.use_boardom_arguments()
    #      other_cfg = bd.Config()
    #      other_cfg.use_boardom_arguments('core')
    #      bd.cfg.setup()
    #      other_cfg.setup(use_sysargv=False)
    #      assert 'session_name' in bd.cfg
    #      assert 'session_name' not in other_cfg

    def test_can_change_session_name_when_creating_session(self):
        sys.argv = sys.argv + [
            '--session_name',
            'foo_session',
            '--project_path',
            f'{_ASSETS_DIR}',
        ]
        bd.cfg.use_boardom_arguments()
        bd.cfg.create_session(session_name='bar_session')
        assert bd.cfg.session_name == 'bar_session'
        assert bd.cfg.project_path == _ASSETS_DIR
        assert os.path.exists(os.path.join(_ASSETS_DIR, 'bar_session'))

    def test_can_change_session_name_when_creating_session_using_a_function(self):
        sys.argv = sys.argv + [
            '--session_name',
            'foo_session',
            '--project_path',
            f'{_ASSETS_DIR}',
            '--lr',
            '10',
        ]
        bd.cfg.use_boardom_arguments()
        bd.cfg.create_session(session_name=lambda x: x.session_name + str(int(x.lr)))
        assert bd.cfg.session_name == 'foo_session10'
        assert bd.cfg.project_path == _ASSETS_DIR
        assert os.path.exists(os.path.join(_ASSETS_DIR, 'foo_session10'))

    def test_using_empty_argument_list_resets_previously_added_ones(self):
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        bd.cfg.empty_argument_list()
        sys.argv = sys.argv + ['--foo', '3']
        with pytest.raises(SystemExit):
            bd.setup()

    def test_can_not_set_g_or_other_members_as_an_argument(self):
        with pytest.raises(RuntimeError):
            bd.cfg.add_argument(flag='--g')
        with pytest.raises(RuntimeError):
            bd.cfg.add_argument(flag='--add_argument')
        for x in dir(bd.cfg):
            with pytest.raises(RuntimeError):
                bd.cfg.add_argument(flag=f'--{x}')

    def test_use_boardom_arguments_does_not_reset_previously_added_ones(self):
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        bd.cfg.use_boardom_arguments()
        bd.cfg.use_boardom_arguments()
        sys.argv = sys.argv + ['--foo', '3']
        bd.setup()

    def test_adding_boardom_arguments_that_already_exist_prints_warning(self, capsys):
        bd.cfg.add_argument(flag='--lr', default=42, help='Foo')
        bd.cfg.use_boardom_arguments()
        captured = capsys.readouterr()
        assert 'Overriding previously defined argument: lr' in captured.out
        assert bd.cfg.lr == 1e-3

    def test_can_use_config_file_in_command_line(self):
        simple_cfg_file = create_asset("--foo 3", '.bd')
        sys.argv = sys.argv + [simple_cfg_file]
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        assert bd.cfg.foo == '3'

    def test_can_provide_cfg_file_to_setup(self):
        simple_cfg_file = create_asset("--foo 3", '.bd')
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        bd.cfg.setup(simple_cfg_file)
        assert bd.cfg.foo == '3'

    def test_non_existing_cfg_file_raises_error(self):
        simple_cfg_file = os.path.join(_ASSETS_DIR, 'does_not_exist.bd')
        sys.argv = sys.argv + [simple_cfg_file]
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        with pytest.raises(SystemExit):
            bd.setup()

    def test_can_use_multiple_config_files(self):
        f_cfg_1 = create_asset("foo 3", '.bd')
        f_cfg_2 = create_asset("bar 5", '.bd')
        sys.argv = sys.argv + [f_cfg_1, f_cfg_2]
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        bd.cfg.add_argument(flag='--bar', default=1, help='Foo')
        assert bd.cfg.foo == '3'
        assert bd.cfg.bar == '5'

    def test_setup_provided_cfg_files_override_cli_config_files(self):
        cli_cfg_file_1 = create_asset("--foo 3", '.bd')
        cli_cfg_file_2 = create_asset("--foo 5", '.bd')
        setup_provided_file = create_asset("--foo 7", '.bd')
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        sys.argv = sys.argv + [cli_cfg_file_1, cli_cfg_file_2]
        bd.cfg.setup(setup_provided_file)
        assert bd.cfg.foo == '7'

    def test_latest_setup_provided_cfg_files_override_previous_ones(self):
        cli_cfg_file_1 = create_asset("--foo 3", '.bd')
        cli_cfg_file_2 = create_asset("--foo 5", '.bd')
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        bd.cfg.setup(cli_cfg_file_1, cli_cfg_file_2)
        assert bd.cfg.foo == '5'

    def test_only_supported_extensions_work(self):
        f_cfg_1 = create_asset("foo 3", '.bdx')
        sys.argv = sys.argv + [f_cfg_1]
        with pytest.raises(SystemExit):
            bd.setup()

    def test_can_get_argument_with_function_attribute_and_item(self):
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        assert bd.cfg.get('foo') == 42
        assert bd.cfg.foo == 42
        assert bd.cfg['foo'] == 42

    def test_can_not_get_undefined_argument(self):
        from boardom.config.config import ConfigGetError

        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        with pytest.raises(ConfigGetError):
            bd.cfg.get('bar')
        with pytest.raises(AttributeError):
            bd.cfg.bar
        with pytest.raises(KeyError):
            bd.cfg['bar']

    def test_get_function_default_works(self):
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        assert bd.cfg.get('foo') == 42
        assert bd.cfg.get('bar', default=None) is None
        assert bd.cfg.get('bar', default=3) == 3

    def test_can_set_argument_value_with_function_attribute_and_item(self):
        bd.cfg.add_argument(flag='--foo', default=42, help='Foo')
        assert bd.cfg.foo == 42
        bd.cfg.set('foo', 3)
        assert bd.cfg.foo == 3
        bd.cfg.foo = 5
        assert bd.cfg.foo == 5
        bd.cfg['foo'] = 7
        assert bd.cfg.foo == 7

    def test_can_not_set_undefined_argument(self):
        from boardom.config.config import ConfigSetError

        bd.cfg.add_argument(flag='--foo', default=42)
        with pytest.raises(ConfigSetError):
            bd.cfg.set('bar', 5)
        with pytest.raises(AttributeError):
            bd.cfg.bar = 10
        with pytest.raises(KeyError):
            bd.cfg['bar'] = 10

    def test_can_not_set_untouchables(self):
        from boardom.config.config import ConfigSetError
        from boardom.config.common import UNTOUCHABLES

        for unt in UNTOUCHABLES:
            with pytest.raises(ConfigSetError):
                bd.cfg.set(unt, 10)

    def test_can_define_group_in_config_file_and_access_group_using_g_property(self):
        simple_cfg_file = create_asset(r"foo foo_a {a}", '.bd')
        sys.argv = sys.argv + [simple_cfg_file]
        bd.cfg.add_argument(flag='--foo', default='def_foo')
        assert bd.cfg.foo == 'def_foo'
        assert bd.cfg.g.a.foo == 'foo_a'
        assert bd.cfg.foo == 'def_foo'

    def test_g_property_works_as_a_context_manager(self):
        simple_cfg_file = create_asset(r"foo foo_a {a}", '.bd')
        sys.argv = sys.argv + [simple_cfg_file]
        bd.cfg.add_argument(flag='--foo', default='def_foo')
        grp_a = bd.cfg.g.a
        assert bd.cfg.foo == 'def_foo'
        with grp_a:
            assert bd.cfg.foo == 'foo_a'
            assert grp_a.foo == 'foo_a'
            assert bd.cfg.g.a.foo == 'foo_a'
        assert bd.cfg.foo == 'def_foo'

    def test_can_not_retrieve_default_value_from_group_context_if_no_group_given(self):
        simple_cfg_file = create_asset(r"foo foo_a {a}", '.bd')
        sys.argv = sys.argv + [simple_cfg_file]
        bd.cfg.add_argument(flag='--foo', default='foo_def')
        bd.cfg.add_argument(flag='--bar', default='bar_def')

        grp_a = bd.cfg.g.a
        assert bd.cfg.bar == 'bar_def'
        assert bd.cfg.foo == 'foo_def'
        with grp_a:
            assert bd.cfg.foo == 'foo_a'
            with pytest.raises(AttributeError):
                bd.cfg.bar
        assert bd.cfg.bar == 'bar_def'
        assert bd.cfg.foo == 'foo_def'

    def test_can_retrieve_default_value_from_group_context_with_dg(self):
        simple_cfg_file = create_asset(r"foo foo_a {a}", '.bd')
        sys.argv = sys.argv + [simple_cfg_file]
        bd.cfg.add_argument(flag='--foo', default='foo_def')
        bd.cfg.add_argument(flag='--bar', default='bar_def')

        grp_a = bd.cfg.g.a
        assert bd.cfg.bar == 'bar_def'
        assert bd.cfg.foo == 'foo_def'
        with grp_a:
            assert bd.cfg.foo == 'foo_a'
            # Crucial bit is here
            assert bd.cfg.dg.bar == 'bar_def'
            assert bd.cfg.dg.foo == 'foo_def'

            assert bd.cfg.foo == 'foo_a'
            with pytest.raises(AttributeError):
                bd.cfg.bar

        assert bd.cfg.bar == 'bar_def'
        assert bd.cfg.foo == 'foo_def'

    def test_can_nest_groups(self):
        from boardom import cfg

        simple_cfg_file = create_asset(
            r"""
            {a}
                foo foo_a
                bar bar_a
                bar bar_ab {b}
            {}
            foo foo_b {b}
            """,
            '.bd',
        )
        sys.argv = sys.argv + [simple_cfg_file]
        cfg.add_argument(flag='--foo', default='foo_def')
        cfg.add_argument(flag='--bar', default='bar_def')

        # Check the defaults
        assert bd.cfg.foo == 'foo_def'
        assert bd.cfg.bar == 'bar_def'

        # Get groups
        grp_a = bd.cfg.g.a
        grp_ab = grp_a.g.b
        grp_b = bd.cfg.g.b

        # Group a
        assert grp_a.foo == 'foo_a'
        assert grp_a.bar == 'bar_a'

        # Group b
        assert grp_b.foo == 'foo_b'
        with pytest.raises(AttributeError):
            grp_b.bar

        # Group ab
        with pytest.raises(AttributeError):
            grp_ab.foo
        assert grp_ab.bar == 'bar_ab'

        # Check that we can retrieve the defaults again
        assert bd.cfg.foo == 'foo_def'
        assert bd.cfg.bar == 'bar_def'

        with grp_a:
            grp_b_created_in_grp_a_ctx = bd.cfg.g.b
            # Check the defaults now changed in context
            assert bd.cfg.foo == 'foo_a'
            assert bd.cfg.bar == 'bar_a'

            assert grp_a.foo == 'foo_a'
            assert grp_a.bar == 'bar_a'

            # Group b now also has a because we are in the context
            with pytest.raises(AttributeError):
                grp_b.foo
            assert grp_b.bar == 'bar_ab'

            with pytest.raises(AttributeError):
                grp_ab.foo
            assert grp_ab.bar == 'bar_ab'

            # Group b now also has a because we are in the context
            with pytest.raises(AttributeError):
                grp_b_created_in_grp_a_ctx.foo
            assert grp_b_created_in_grp_a_ctx.bar == 'bar_ab'

        # Check that we can retrieve the defaults again
        assert bd.cfg.foo == 'foo_def'
        assert bd.cfg.bar == 'bar_def'

        # Check all the groups behave as before the context manager
        # Group a
        assert grp_a.foo == 'foo_a'
        assert grp_a.bar == 'bar_a'

        # Group b
        assert grp_b.foo == 'foo_b'
        with pytest.raises(AttributeError):
            grp_b.bar

        # Group ab
        with pytest.raises(AttributeError):
            grp_ab.foo
        assert grp_ab.bar == 'bar_ab'

        # Check that we can retrieve the defaults again
        assert bd.cfg.foo == 'foo_def'
        assert bd.cfg.bar == 'bar_def'

        # Check that the group b created in the context manager is just group b now
        assert grp_b_created_in_grp_a_ctx.foo == 'foo_b'
        with pytest.raises(AttributeError):
            grp_b_created_in_grp_a_ctx.bar

        # Now check grp_b context manager
        with grp_b:
            # Check the defaults now changed in context
            assert bd.cfg.foo == 'foo_b'
            with pytest.raises(AttributeError):
                bd.cfg.bar

            # Group a now should be ab in context
            with pytest.raises(AttributeError):
                grp_a.foo
            assert grp_a.bar == 'bar_ab'

            # grp_b is just grp_b
            assert grp_b.foo == 'foo_b'
            with pytest.raises(AttributeError):
                grp_b.bar

            # ab is ab
            with pytest.raises(AttributeError):
                grp_ab.foo
            assert grp_ab.bar == 'bar_ab'

            # Group b created in a context is just b now
            assert grp_b_created_in_grp_a_ctx.foo == 'foo_b'
            with pytest.raises(AttributeError):
                grp_b_created_in_grp_a_ctx.bar

        # Check ab context manager
        with grp_ab:
            # all are ab now
            with pytest.raises(AttributeError):
                bd.cfg.foo
            assert bd.cfg.bar == 'bar_ab'

            with pytest.raises(AttributeError):
                grp_a.foo
            assert grp_a.bar == 'bar_ab'

            with pytest.raises(AttributeError):
                grp_b.foo
            assert grp_b.bar == 'bar_ab'

            with pytest.raises(AttributeError):
                grp_ab.foo
            assert grp_ab.bar == 'bar_ab'

            with pytest.raises(AttributeError):
                grp_b_created_in_grp_a_ctx.foo
            assert grp_b_created_in_grp_a_ctx.bar == 'bar_ab'

        # Check that we can retrieve the defaults again
        assert bd.cfg.foo == 'foo_def'
        assert bd.cfg.bar == 'bar_def'

        # Check all the groups behave as before the context manager
        # Group a
        assert grp_a.foo == 'foo_a'
        assert grp_a.bar == 'bar_a'

        # Group b
        assert grp_b.foo == 'foo_b'
        with pytest.raises(AttributeError):
            grp_b.bar

        # Group ab
        with pytest.raises(AttributeError):
            grp_ab.foo
        assert grp_ab.bar == 'bar_ab'

        # Check that we can retrieve the defaults again
        assert bd.cfg.foo == 'foo_def'
        assert bd.cfg.bar == 'bar_def'

        # Check that the group b created in the context manager is just group b now
        assert grp_b_created_in_grp_a_ctx.foo == 'foo_b'
        with pytest.raises(AttributeError):
            grp_b_created_in_grp_a_ctx.bar

    def test_can_retrieve_default_value_from_nested_groups_with_dg(self):
        from boardom import cfg

        simple_cfg_file = create_asset(
            r"""
            {a}
                foo foo_a
                bar bar_a
                bar bar_ab {b}
            {}
            foo foo_b {b}
            """,
            '.bd',
        )
        sys.argv = sys.argv + [simple_cfg_file]
        cfg.add_argument(flag='--foo', default='foo_def')
        cfg.add_argument(flag='--bar', default='bar_def')

        # Get groups
        grp_a = bd.cfg.g.a
        grp_ab = grp_a.g.b
        grp_b = bd.cfg.g.b
        dg = bd.cfg.dg

        assert cfg.dg.foo == 'foo_def'
        assert cfg.dg.bar == 'bar_def'
        assert grp_a.dg.foo == 'foo_def'
        assert grp_a.dg.bar == 'bar_def'
        assert grp_b.dg.foo == 'foo_def'
        assert grp_b.dg.bar == 'bar_def'
        assert grp_ab.dg.foo == 'foo_def'
        assert grp_ab.dg.bar == 'bar_def'
        assert dg.foo == 'foo_def'
        assert dg.bar == 'bar_def'

        with grp_a:
            assert cfg.dg.foo == 'foo_def'
            assert cfg.dg.bar == 'bar_def'
            assert grp_a.dg.foo == 'foo_def'
            assert grp_a.dg.bar == 'bar_def'
            assert grp_b.dg.foo == 'foo_def'
            assert grp_b.dg.bar == 'bar_def'
            assert grp_ab.dg.foo == 'foo_def'
            assert grp_ab.dg.bar == 'bar_def'
            assert dg.foo == 'foo_def'
            assert dg.bar == 'bar_def'

            assert cfg.foo == 'foo_a'
            assert cfg.bar == 'bar_a'

            with grp_b:
                with pytest.raises(AttributeError):
                    cfg.foo
                assert cfg.bar == 'bar_ab'

                assert cfg.dg.foo == 'foo_def'
                assert cfg.dg.bar == 'bar_def'
                assert grp_a.dg.foo == 'foo_def'
                assert grp_a.dg.bar == 'bar_def'
                assert grp_b.dg.foo == 'foo_def'
                assert grp_b.dg.bar == 'bar_def'
                assert grp_ab.dg.foo == 'foo_def'
                assert grp_ab.dg.bar == 'bar_def'
                assert dg.foo == 'foo_def'
                assert dg.bar == 'bar_def'

            with grp_ab:
                with pytest.raises(AttributeError):
                    cfg.foo
                assert cfg.bar == 'bar_ab'

                assert cfg.dg.foo == 'foo_def'
                assert cfg.dg.bar == 'bar_def'
                assert grp_a.dg.foo == 'foo_def'
                assert grp_a.dg.bar == 'bar_def'
                assert grp_b.dg.foo == 'foo_def'
                assert grp_b.dg.bar == 'bar_def'
                assert grp_ab.dg.foo == 'foo_def'
                assert grp_ab.dg.bar == 'bar_def'
                assert dg.foo == 'foo_def'

    def test_fallback_works(self):
        from boardom import cfg

        simple_cfg_file = create_asset(
            r"""
            {a}
                foo foo_a
                {b}
                    bar bar_ab
                    {c}
                        bar bar_abc
                        foo foo_abc
                        {d}
                            bar bar_abcd
                        {}
                    {}
                {}

                {e}
                    foo foo_ae
                    {f}
                        bar bar_aef
                    {}
                {}
            {}
            {b}
                {c}
                    baz baz_bc
                {}
            {}
            baz baz_ac {a, c}
            baz baz_c {c}
            """,
            '.bd',
        )
        sys.argv = sys.argv + [simple_cfg_file]
        cfg.add_argument(flag='--foo', default='foo_def')
        cfg.add_argument(flag='--bar', default='bar_def')
        cfg.add_argument(flag='--baz', default='baz_def')
        grp_a = cfg.g.a
        grp_ab = grp_a.g.b
        grp_abc = grp_ab.g.c
        grp_abcd = grp_abc.g.d
        grp_abcde = grp_abcd.g.e
        grp_abcdef = grp_abcde.g.f

        # Defaults
        assert cfg.foo == 'foo_def'
        assert cfg.bar == 'bar_def'

        # Without fallback
        assert grp_a.foo == 'foo_a'
        # This does not fallback to default
        with pytest.raises(AttributeError):
            grp_a.bar
        # This does not fall back to group a
        with pytest.raises(AttributeError):
            grp_ab.foo
        assert grp_ab.bar == 'bar_ab'
        assert grp_abc.foo == 'foo_abc'
        assert grp_abc.bar == 'bar_abc'
        with pytest.raises(AttributeError):
            grp_abcd.foo
        assert grp_abcd.bar == 'bar_abcd'
        with pytest.raises(AttributeError):
            grp_abcde.bar
        with pytest.raises(AttributeError):
            grp_abcde.foo
        with pytest.raises(AttributeError):
            grp_abcdef.foo
        with pytest.raises(AttributeError):
            grp_abcdef.bar

        with pytest.raises(AttributeError):
            grp_a.baz
        with pytest.raises(AttributeError):
            grp_ab.baz
        with pytest.raises(AttributeError):
            grp_abc.baz
        with pytest.raises(AttributeError):
            grp_abcd.baz
        with pytest.raises(AttributeError):
            grp_abcde.baz
        with pytest.raises(AttributeError):
            grp_abcdef.baz

        assert cfg.g.c.baz == 'baz_c'
        assert cfg.g.b.g.c.baz == 'baz_bc'
        assert cfg.g.a.g.c.baz == 'baz_ac'

        # With fallback
        with cfg.group_fallback():
            # Defaults
            assert cfg.foo == 'foo_def'
            assert cfg.bar == 'bar_def'

            assert grp_a.foo == 'foo_a'
            assert grp_a.bar == 'bar_def'
            assert grp_ab.foo == 'foo_a'
            assert grp_ab.bar == 'bar_ab'
            assert grp_abc.foo == 'foo_abc'
            assert grp_abc.bar == 'bar_abc'
            assert grp_abcd.foo == 'foo_abc'
            assert grp_abcd.bar == 'bar_abcd'
            # IMPORTANT: Here it falls back to foo_ae and not foo_abc
            assert grp_abcde.foo == 'foo_ae'
            assert grp_abcde.bar == 'bar_abcd'
            assert grp_abcdef.foo == 'foo_ae'
            assert grp_abcdef.bar == 'bar_aef'

            assert grp_a.baz == 'baz_def'
            assert grp_ab.baz == 'baz_def'
            assert grp_abc.baz == 'baz_bc'
            assert grp_abcd.baz == 'baz_bc'
            assert grp_abcde.baz == 'baz_bc'
            assert grp_abcdef.baz == 'baz_bc'

            # Check that we can get nested defaults with fallback enabled
            with grp_abcde:
                assert cfg.dg.foo == 'foo_def'
                assert cfg.dg.bar == 'bar_def'
                assert cfg.dg.baz == 'baz_def'
                assert grp_abcde.dg.foo == 'foo_def'
                assert grp_abc.dg.bar == 'bar_def'
                assert grp_a.dg.baz == 'baz_def'

    def test_get_default_works(self):
        from boardom import cfg

        simple_cfg_file = create_asset(
            r"""
            foo foo_a {a}
            """,
            '.bd',
        )
        sys.argv = sys.argv + [simple_cfg_file]
        cfg.add_argument(flag='--foo', default='foo_def')
        cfg.add_argument(flag='--bar', default='bar_def')

        assert cfg.get('foo') == 'foo_def'
        assert cfg.get('foo', None) == 'foo_def'
        assert cfg.get('bar') == 'bar_def'
        assert cfg.get('bar', None) == 'bar_def'
        with pytest.raises(RuntimeError):
            cfg.get('baz')
        assert cfg.get('baz', None) is None

        grp_a = cfg.g.a

        assert grp_a.get('foo') == 'foo_a'
        assert grp_a.get('foo', None) == 'foo_a'
        with pytest.raises(RuntimeError):
            grp_a.get('bar')
        assert grp_a.get('bar', None) is None
        with pytest.raises(RuntimeError):
            grp_a.get('baz')
        assert grp_a.get('baz', None) is None
