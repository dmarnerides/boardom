import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LambdaLR
import boardom as bd


@bd.autoconfig(ignore='parameters')
def optimizer(
    parameters,
    optimizer='adam',
    lr=1e-3,
    momentum=0.9,
    dampening=0.0,
    beta1=0.9,
    beta2=0.999,
    rho=0.9,
    alpha=0.99,
    centered=False,
    lr_decay=0.0,
    optim_eps=1e-8,
    weight_decay=0.0,
):
    """Returns the optimizer for the given model.

    Args:
        model (nn.Module): The network for the optimizer.
        extra_params (generator, optional): Extra parameters to pass to the optimizer.
        subset (string, optional): Specifies the subset of the relevant
            categories, if any of them was split (default, None).

    Relevant Command Line Arguments:

        - **optimizer**: `--optimizer`, `--lr`, `--momentum`,
            `--dampening`, `--beta1`, `--beta2`, `--weight_decay`.

    Note:
        Settings are automatically acquired from a call to :func:`boardom.config.parse`
        from the built-in ones. If :func:`boardom.config.parse` was not called in the
        main script, this function will call it.
    """

    optimizer = optimizer.lower()
    if optimizer == 'adam':
        ret_optimizer = torch.optim.Adam(
            parameters, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
        )
    elif optimizer == 'adamw':
        ret_optimizer = torch.optim.AdamW(
            parameters, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
        )
    elif optimizer == 'sgd':
        ret_optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
        )
    elif optimizer == 'adadelta':
        ret_optimizer = torch.optim.Adadelta(
            parameters,
            lr=lr,
            rho=rho,
            eps=optim_eps,
            weight_decay=weight_decay,
        )
    elif optimizer == 'adagrad':
        ret_optimizer = torch.optim.Adagrad(
            parameters, lr=lr, lr_decay=lr_decay, weight_decay=weight_decay
        )
    elif optimizer == 'sparseadam':
        ret_optimizer = torch.optim.SparseAdam(
            parameters, lr=lr, betas=(beta1, beta2), eps=optim_eps
        )
    elif optimizer == 'adamax':
        ret_optimizer = torch.optim.Adamax(
            parameters,
            lr=lr,
            betas=(beta1, beta2),
            eps=optim_eps,
            weight_decay=weight_decay,
        )
    elif optimizer == 'rmsprop':
        ret_optimizer = torch.optim.RMSprop(
            parameters,
            lr=lr,
            alpha=alpha,
            eps=optim_eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
    else:
        raise NotImplementedError(f'Optimizer {optimizer} not implemented.')
    return ret_optimizer


@bd.autoconfig(ignore='optimizer')
def scheduler(
    optimizer,
    lr_schedule='none',
    lr_step_size=100,
    lr_patience=10,
    lr_cooldown=0,
    lr_min=1e-7,
    lr_ratio=0.5,
):
    """Returns a scheduler callable closure which accepts one argument.

    Configurable using command line arguments.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for the scheduler.
        subset (string, optional): Specifies the subset of the relevant
            categories, if any of them was split (default, None).

    Relevant Command Line Arguments:

        - **scheduler**: `--lr_schedule`, `--lr_step_size`, `--lr_patience`,
            `--lr_cooldown`, `--lr_ratio`, `--lr_min`,

    Note:
        Settings are automatically acquired from a call to :func:`boardom.config.parse`
        from the built-in ones. If :func:`boardom.config.parse` was not called in the
        main script, this function will call it.
    """
    if lr_schedule == 'plateau':
        ret_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=lr_ratio,
            threshold=0.0001,
            patience=lr_patience,
            verbose=True,
            threshold_mode='rel',
            cooldown=lr_cooldown,
            min_lr=lr_min,
            eps=1e-08,
        )
    elif lr_schedule == 'step':
        ret_scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_ratio)
    elif lr_schedule == 'none':
        ret_scheduler = LambdaLR(optimizer, lr_lambda=lambda x: 1)
    else:
        raise ValueError(f'Unknown lr_schedule {lr_schedule}')

    if lr_schedule == 'plateau':

        def schedule_fn(metric):
            ret_scheduler.step(metric)

    else:

        def schedule_fn(metric):
            ret_scheduler.step()

    def schedule_step(metric=None):
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        schedule_fn(metric)
        new_lrs = [group['lr'] for group in optimizer.param_groups]
        for i, (current_lr, new_lr) in enumerate(zip(current_lrs, new_lrs)):
            if new_lr != current_lr:
                bd.log(
                    f'Learning rate changed from {current_lr:.2e} to {new_lr:.2e} (param_group {i})'
                )

    return schedule_step
