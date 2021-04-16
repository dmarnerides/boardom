import torch
from boardom import str2bool, process_path
import torchvision as tv


# MODEL CONFIG
_UPSAMPLE_TYPES = ['transpose', 'nearest', 'bilinear']
_DOWNSAMPLE_TYPES = [
    'strided',
    'nearest',
    'bilinear',
    'strided_preact_bbn',
    'nearest_preact_bbn',
    'bilinear_preact_bbn',
]
_BLOCK_TYPES = [
    'a',
    'n',
    'c',
    'cna',
    'nac',
    'r_orig',
    'r_preact',
    'r_preact_c',
    'r_preact_bbn',
]
_UNET_FUSION_TYPES = ['lms_guided_filter', 'guided_filter', 'cat', 'add']
_UNET_TYPES = ['original', 'old_gunet', 'custom', 'pretrained']

# The lists here (temporarily) hold the elements defined in the
# configuration files
# They hold the elements per line as they are parsed.


def _create_datum(value, groups=None, tags=None, meta=None):
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise RuntimeError('Meta property must be a dictionary.')
    if tags is None:
        tags = []
    if not isinstance(tags, list):
        raise RuntimeError('Tags property must be a list.')
    if groups is None:
        group = Group()
    else:
        group = Group(groups)
    return {group: {'value': value, 'tags': tags, 'meta': meta}}


class Group(set):
    _SEPARATOR = '.'
    _DEFAULT_STR = 'default_grp'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if Group._DEFAULT_STR in self:
            self.remove(Group._DEFAULT_STR)
        if None in self:
            self.remove(None)

    def add(self, key):
        if (key is None) or (key == Group._DEFAULT_STR):
            return
        super().add(key)

    def __str__(self):
        lst = list(self)
        lst.sort()
        return Group._SEPARATOR.join(lst)

    @property
    def is_default(self):
        return len(self) == 0

    @staticmethod
    def from_full_argname(arg_name):
        arg_name, *groups = arg_name.split(Group._SEPARATOR)
        return arg_name, Group(groups)

    def build_full_argname(self, arg_name):
        if self.is_default:
            return arg_name
        return f'{arg_name}{Group._SEPARATOR}{self}'

    def __hash__(self):
        return hash(str(self))


def _is_valid_argname(x):
    return isinstance(x, str) and x.isidentifier() and (not x.startswith('_'))


OPTIMIZERS = [
    'adam',
    'adamw',
    'sgd',
    'adadelta',
    'adagrad',
    'sparseadam',
    'adamax',
    'rmsprop',
]

DEVICE_KEYS = ['device', 'cudnn_benchmark']
CRITERIA_KEYS = ['criteria', 'criterion_weight']
OPTIMIZER_KEYS = [
    'optimizer',
    'lr',
    'momentum',
    'dampening',
    'beta1',
    'beta2',
    'rho',
    'alpha',
    'centered',
    'lr_decay',
    'weight_decay',
    'find_good_lr',
]
DATALOADER_KEYS = [
    'num_workers',
    'batch_size',
    'shuffle',
    'pin_memory',
    'drop_last',
    'timeout',
    'prefetch_factor',
    'persistent_workers',
]
CHECKPOINT_KEYS = ['overwrite', 'strict', 'use_timestamps']
IMAGE_SAMPLER_KEYS = ['overwrite', 'use_timestamps', 'extension']


TORCHVISION_DATASETS = [
    'mnist',
    'fashionmnist',
    'cifar10',
    'cifar100',
]


AUTOMATIC_ARGS = [
    'process_id',
    'time_configured',
    'session_path',
]

CORE_SETTINGS = [
    dict(
        flag='--create_session',
        type=str2bool,
        default=False,
        help='Save session data in the .session.bd.json file.',
    ),
    dict(
        flag='--project_path',
        type=process_path,
        default='.',
        help='Root directory for placing session sub-directories.',
    ),
    dict(
        flag='--session_name',
        default='bd_session',
        help='Name of session',
    ),
    dict(
        flag='--log_stdout',
        type=str2bool,
        default=False,
        help=('Output all stdout to a log file.'),
    ),
    dict(
        flag='--copy_config_files',
        type=str2bool,
        default=False,
        help=(
            'Copy configuration files (.bd files)' ' used when launching main script.'
        ),
    ),
    dict(
        flag='--print_cfg',
        type=str2bool,
        default=False,
        help='Print configuration when setup() is done.',
    ),
    dict(
        flag='--save_full_config',
        type=str2bool,
        default=False,
        help='Save full configuration when setup() is done.',
    ),
    dict(
        flag='--log_csv',
        type=str2bool,
        default=False,
        help='Log stuff in csv files.',
    ),
    dict(
        flag='--log_tensorboard',
        type=str2bool,
        default=False,
        help='Use tensorboard.',
    ),
    dict(flag='--log_boardom', type=str2bool, default=False, help='Use boardom.'),
    dict(
        flag='--autocommit',
        type=str2bool,
        default=False,
        help=('Autocommit on a separate branch'),
    ),
    dict(
        flag='--only_run_same_hash',
        type=str2bool,
        default=False,
        help=(
            'Only run code that matches the previous automatically generated git hash.'
        ),
    ),
]

EXTRA_SETTINGS = [
    dict(flag='--train', type=str2bool, default=True, help='Do training.'),
    dict(flag='--validate', type=str2bool, default=False),
    dict(flag='--test', type=str2bool, default=False, help='Do testing.'),
    dict(
        flag='--max_epochs',
        type=int,
        default=1000,
        help='Maximum number of epochs',
    ),
    # Frequencies default to -1 such that they are not unintentionally used
    dict(
        flag='--per_step',
        type=int,
        default=-1,
    ),
    dict(
        flag='--per_epoch',
        type=int,
        default=-1,
    ),
    dict(
        flag='--per_minute',
        type=float,
        default=-1,
    ),
    dict(
        flag='--timestamp',
        type=str2bool,
        default=True,
    ),
    dict(flag='--device', type=torch.device, default='cpu', help='Device to use'),
    dict(
        flag='--cudnn_benchmark',
        type=str2bool,
        default=False,
        help='Use cudnn benchmark mode',
    ),
    dict(
        flag='--criteria',
        nargs='+',
        default=[],
        help='Criteria to use',
    ),
    dict(
        flag='--criterion_weight',
        type=float,
        default=1.0,
        help='Weights for criteria.',
    ),
    dict(
        flag='--metrics',
        nargs='+',
        default=[],
        help='Criteria to use',
    ),
    dict(
        flag='--optimizer',
        type=str.lower,
        choices=OPTIMIZERS,
        default='adam',
        help='Optimizer',
    ),
    dict(flag='--lr', type=float, default=1e-3, help='Learning rate'),
    dict(flag='--momentum', type=float, default=0.9, help='SGD Momentum'),
    dict(flag='--dampening', type=float, default=0.0, help='SGD Dampening'),
    dict(flag='--beta1', type=float, default=0.9, help='Adam beta1 parameter'),
    dict(flag='--beta2', type=float, default=0.999, help='Adam beta2 parameter'),
    dict(flag='--rho', type=float, default=0.9, help='Adadelta rho parameter'),
    dict(
        flag='--alpha',
        type=float,
        default=0.99,
        help='RMSprop alpha parameter',
    ),
    dict(
        flag='--centered',
        type=str2bool,
        default=False,
        help='RMSprop centered flag',
    ),
    dict(flag='--lr_decay', type=float, default=0.0, help='Adagrad lr_decay'),
    dict(
        flag='--optim_eps',
        type=float,
        default=1e-8,
        help='Term added to denominator for numerical stability.',
    ),
    dict(
        flag='--weight_decay',
        type=float,
        default=0.0,
        help='Weight decay / L2 regularization.',
    ),
    dict(
        flag='--find_good_lr',
        type=str2bool,
        default=False,
        help='Find best lr',
    ),
    dict(
        flag='--num_workers',
        type=int,
        default=0,
        help='Number of data loading threads',
    ),
    dict(flag='--batch_size', type=int, default=1, help='Batch size for loader'),
    dict(
        flag='--shuffle',
        type=str2bool,
        default=True,
        help='Loader shuffles data each epoch',
    ),
    dict(
        flag='--pin_memory',
        type=str2bool,
        default=False,
        help='Pin tensor memory for efficient GPU loading',
    ),
    dict(
        flag='--drop_last',
        type=str2bool,
        default=False,
        help='Drop last batch if its size is less than batch size',
    ),
    dict(
        flag='--timeout',
        type=float,
        default=0,
        help='Timeout for data loader.',
    ),
    dict(flag='--prefetch_factor', type=int, default=2),
    dict(flag='--persistent_workers', type=str2bool, default=False),
    dict(flag='--overwrite', type=str2bool, default=True),
    dict(flag='--strict', type=str2bool, default=True),
    dict(flag='--use_timestamps', type=str2bool, default=True),
    dict(flag='--save_state_dicts', type=str2bool, default=True),
    dict(flag='--display', type=str2bool, default=False),
    dict(flag='--save', type=str2bool, default=False),
    dict(
        flag='--seed',
        type=int,
        default=None,
        help='Seed for random number generation.',
    ),
    dict(
        flag='--data_root_path', type=process_path, default='.', help='Data directory'
    ),
    dict(flag='--dataset', default=None, help='Dataset in use.'),
    dict(
        flag='--torchvision_dataset',
        type=str.lower,
        choices=TORCHVISION_DATASETS,
        default=None,
        help='Specific dataset to use',
    ),
    dict(
        flag='--download',
        type=str2bool,
        default=True,
        help='Download the dataset.',
    ),
    dict(
        flag='--grow_dataset',
        type=int,
        default=1,
        help='Growth factor for bd.GrowDataset.',
    ),
    dict(
        flag='--load_in_memory',
        type=str2bool,
        default=False,
        help='Load data in memory',
    ),
    dict(
        flag='--load_encoded',
        type=str2bool,
        default=False,
        help='Whether to load encoded',
    ),
    dict(
        flag='--encoded_positions',
        type=int,
        nargs='+',
        default=[0],
        help='Positions of images that the dataset returns.',
    ),
    dict(
        flag='--compress_loaded',
        type=int,
        default=0,
        help=(
            'Compress Loaded datasets (if using LoadedDataset). '
            '[0-9] with 0 being no compression.'
        ),
    ),
    dict(
        flag='--data_extensions',
        nargs='+',
        default=['jpg', 'png', 'hdr', 'exr', 'pfm'],
        help='Extensions of data to load.',
    ),
    dict(
        flag='--lr_schedule',
        choices=['plateau', 'step', 'none'],
        default='step',
        help='Learning rate schedule',
    ),
    dict(
        flag='--lr_step_size',
        type=int,
        default=100,
        help='Epochs per learning rate decrease (step).',
    ),
    dict(
        flag='--lr_patience',
        type=int,
        default=10,
        help='Epochs of patience for metric (plateau).',
    ),
    dict(
        flag='--lr_cooldown',
        type=int,
        default=0,
        help='Epochs of cooldown period after lr change (plateau).',
    ),
    dict(
        flag='--lr_min',
        type=float,
        default=1e-7,
        help='Minimum learning rate (plateau)',
    ),
    dict(
        flag='--lr_ratio',
        type=float,
        default=0.5,
        help='Ratio to decrease learning rate by (all)',
    ),
    dict(
        flag='--dataparallel',
        type=int,
        nargs='+',
        default=None,
        help='Use dataparallel module.',
    ),
    dict(
        flag='--normalization',
        default='batchnorm2d',
        help='Normalization module.',
    ),
    dict(
        flag='--activation',
        default='relu',
        help='Activation module.',
    ),
    dict(
        flag='--network_name',
        default='resnet18',
        choices=tv.models.resnet.__all__,
        help='Pretrained resnet network name.',
    ),
    dict(
        flag='--num_pretrained_layers',
        type=int,
        default=5,
        help='Number of pretrained layers for PretrainedResnet.',
    ),
    dict(
        flag='--freeze_batchnorm',
        type=str2bool,
        default=True,
        help='Freeze batch normalization for pretrained networks.',
    ),
    dict(
        flag='--split_before_relus',
        type=str2bool,
        default=False,
        help='Skip encoder features before relus are evaluated for pretrained resnets.',
    ),
    dict(
        flag='--leaky_relu_slope',
        type=float,
        default=0.01,
        help='Slope for leaky relu activation.',
    ),
    dict(
        flag='--kernel_size',
        type=int,
        default=3,
        help='Kernel size.',
    ),
    dict(
        flag='--kernel_sizes',
        nargs='*',
        type=int,
        default=[3],
        help='Kernel sizes.',
    ),
    dict(
        flag='--downsample',
        default='strided',
        choices=_DOWNSAMPLE_TYPES,
        help='Downsampling type.',
    ),
    dict(
        flag='--upsample',
        default='transpose',
        choices=_UPSAMPLE_TYPES,
        help='Upsampling type.',
    ),
    dict(
        flag='--fusion', default='cat', choices=_UNET_FUSION_TYPES, help='Fusion type.'
    ),
    dict(flag='--epsilon', type=float, default=1e-3, help='Generic epsilon value.'),
    dict(
        flag='--learn_epsilon',
        type=str2bool,
        default=True,
        help='LearnEpsilonValue.',
    ),
    dict(
        flag='--bottleneck_gf_adapter',
        type=str2bool,
        default=True,
        help='Bottleck gf adapter for large channel sizes by a factor of 4.',
    ),
    dict(
        flag='--grouped_gf',
        type=str2bool,
        default=False,
        help='Grouped multiplication for guided filter',
    ),
    dict(
        flag='--norm_eps',
        type=float,
        default=1e-5,
        help='Normalization eps value. (BN default)',
    ),
    dict(
        flag='--norm_momentum',
        type=float,
        default=0.1,
        help='Normalization momentum value. (BN default)',
    ),
    dict(
        flag='--norm_affine',
        type=str2bool,
        default=True,
        help='Normalization affine value. (BN default)',
    ),
    dict(
        flag='--norm_track_running_stats',
        type=str2bool,
        default=True,
        help='Normalization track running stats value. (BN default)',
    ),
    dict(
        flag='--hidden_features',
        type=int,
        nargs='+',
        default=[64, 128, 256, 512],
        help='Hidden features.',
    ),
    dict(flag='--in_features', type=int, default=3, help='Input features.'),
    dict(flag='--out_features', type=int, default=3, help='Output features.'),
    dict(
        flag='--block_types',
        nargs='+',
        default=['r_preact'],
        choices=_BLOCK_TYPES,
        help='Type of blocks.',
    ),
    dict(flag='--unet_type', default='custom', choices=_UNET_TYPES, help='UNet type'),
    dict(
        flag='--first_conv_then_resize',
        type=str2bool,
        default=False,
        help='First do convolution and then resize.',
    ),
    dict(flag='--initializer', default='kaimingnormal', help='Initializer'),
    dict(flag='--uniform_a', type=float, default=0, help='a value for uniform init.'),
    dict(flag='--uniform_b', type=float, default=1, help='b value for uniform init.'),
    dict(
        flag='--normal_mean',
        type=float,
        default=0,
        help='mean value for normal init.',
    ),
    dict(
        flag='--normal_std',
        type=float,
        default=1,
        help='std value for normal init.',
    ),
    dict(
        flag='--constant_val',
        type=float,
        default=0,
        help='val value for constant init.',
    ),
    dict(
        flag='--dirac_groups',
        type=int,
        default=1,
        help='groups value for dirac init',
    ),
    dict(
        flag='--xavier_gain',
        type=float,
        default=1,
        help='gain value for xavier init',
    ),
    dict(
        flag='--kaiming_a',
        type=float,
        default=0,
        help='a value for kaiming init',
    ),
    dict(
        flag='--kaiming_mode',
        default='fan_in',
        choices=['fan_in', 'fan_out'],
        help='mode for kaiming init',
    ),
    dict(
        flag='--kaiming_nonlinearity',
        default='leaky_relu',
        choices=['relu', 'leaky_relu'],
        help='nonlinearity for kaiming init',
    ),
    dict(
        flag='--orthogonal_gain',
        type=float,
        default=1.0,
        help='gain for orthogonal init',
    ),
    dict(
        flag='--sparse_sparsity',
        type=float,
        default=1.0,
        help='sparsity for sparse init',
    ),
    dict(
        flag='--sparse_std',
        type=float,
        default=0.01,
        help='std for sparse init',
    ),
    dict(flag='--extension', default='.jpg'),
    dict(
        flag='--log_all_iterations',
        type=str2bool,
        default=True,
        help='Log all iterations.',
    ),
    dict(
        flag='--log_averages',
        type=str2bool,
        default=True,
        help='Log averages.',
    ),
]

CORE_ARGNAMES = [x['flag'][2:] for x in CORE_SETTINGS]

DEFAULT_CFG_DICT = {'core': CORE_SETTINGS, 'extra': EXTRA_SETTINGS}

# UNTOUCHABLEs cannot be added to categories or groups
# and can't be changed after setup
UNTOUCHABLES = AUTOMATIC_ARGS + CORE_ARGNAMES
