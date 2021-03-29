from collections.abc import Iterable, Mapping
import os
import matplotlib.pyplot as plt
import cv2
import torch
import pandas as pd
import boardom as bd


def _get_tensor(x):
    x = x[0] if torch.typename(x) in ['tuple', 'list'] else x
    return x


def _get_tensor_dict(x, tag):
    if x is None:
        yield {}
    elif bd.is_tensor(x):
        yield {tag: x}
    elif isinstance(x, Iterable):
        for i, y in enumerate(x):
            for obj in _get_tensor_dict(y, f'{tag}_{i}'):
                yield obj
    elif isinstance(x, Mapping):
        for k, v in x.items():
            for obj in _get_tensor_dict(v, f'{tag}_{k}'):
                yield obj


def save_image(img, title):
    to_save = ((img / img.max()) * 255).astype(int)
    cv2.imwrite('{0}.png'.format(title), to_save)


def process_none(x):
    if x is None:
        x = []
    elif not any((isinstance(x, y) for y in [list, tuple])):
        x = [x]
    return x


def _register(net, hook, modules=None, match_names=None, do_forward=True):
    modules = process_none(modules)
    match_names = process_none(match_names)
    for mod_name, mod in net.named_modules():
        name_match = any(
            torch.typename(modules).find(x) >= 0 for x in match_names
        )
        instance_match = any(isinstance(mod, x) for x in modules)
        if instance_match or name_match:
            if do_forward:
                mod.register_forward_hook(hook(mod_name))
            else:
                mod.register_backward_hook(hook(mod_name))
    return net


def _hook_generator(
    do_input=False,
    do_output=True,
    tag='',
    save_path='.',
    replace=True,
    histogram=True,
    bins=100,
    mode='forward',
    param_names=None,
    rename_fn=None,
    ext='.jpg',
):
    save_path = bd.process_path(save_path, True)
    tensor_names = (
        ['in', 'out']
        if mode in ['forward', 'parameters']
        else ['grad_in', 'grad_out']
    )

    def get_hook(module_name):
        counter = 1

        def hook(module, inp=None, out=None):
            nonlocal counter, tensor_names
            rename_module = rename_fn or bd.identity
            added_tag = tag if tag == '' else tag + '_'
            if mode == 'parameters':
                tensors = {
                    x: _get_tensor(getattr(module, x)) for x in param_names
                }
            else:
                tensor_coll = [
                    (tensor_names[0], inp, do_input),
                    (tensor_names[1], out, do_output),
                ]
                tensors = {}
                for name, x, to_do in tensor_coll:
                    if not to_do:
                        continue
                    for d in _get_tensor_dict(x, name):
                        tensors.update(d)
            for tensor_name, data in tensors.items():
                if data is None:
                    continue
                title_end = '' if replace else '-{0:06d}'.format(counter)
                title_end = title_end + '-hist' if histogram else title_end
                title = f'{added_tag}{rename_module(module_name)}_{tensor_name}{title_end}'
                if histogram:
                    img = bd.torch2cv(data)
                    df = pd.DataFrame(img.reshape(img.size))
                    fig, ax = plt.subplots()
                    df.hist(bins=bins, ax=ax)
                    fig.savefig(os.path.join(save_path, f'{title}{ext}'))
                    plt.close(fig)
                else:
                    if data.dim() > 1:
                        img = bd.torch2cv(bd.make_grid(data))
                        to_save = bd.map_range(img, 0, 255).astype(int)
                        cv2.imwrite(
                            os.path.join(
                                save_path, f'{title}{ext}'.format(title)
                            ),
                            to_save,
                        )
            counter = counter + 1

        return hook

    return get_hook


def forward_hook(
    net,
    modules=None,
    match_names=None,
    do_input=False,
    do_output=True,
    tag='',
    save_path='.',
    replace=True,
    histogram=True,
    bins=100,
    rename_fn=None,
):
    """Registers a forward hook to a network's modules for vizualization of the inputs and outputs.

    When net.forward() is called, the hook saves an image grid or a histogram 
    of input/output of the specified modules.

    Args:
        net (nn.Module): The network whose modules are to be visualized.
        modules (list or tuple, optional): List of class definitions for the
            modules where the hook is attached e.g. nn.Conv2d  (default None).
        match_names (list or tuple, optional): List of strings. If any modules
            contain one of the strings then the hook is attached (default None).
        do_input (bool, optional): If True the input of the module is 
            visualized (default False).
        do_output (bool, optional): If True the output of the module is 
            visualized (default True).
        tag (str, optional): String tag to attach to saved images (default None).
        save_path (str, optional): Path to save visualisation results 
            (default '.').
        replace (bool, optional): If True, the images (from the same module) 
            are replaced whenever the hook is called (default True).
        histogram (bool, optional): If True then the visualization is a
            histrogram, otherwise it's an image grid.
        bins (bool, optional): Number of bins for histogram, if `histogram` is
            True (default 100).

    Note:
        * If modules or match_names are not provided then no hooks will be
          attached.
    """
    hook = _hook_generator(
        do_input,
        do_output,
        tag,
        save_path,
        replace,
        histogram,
        bins,
        'forward',
        rename_fn=rename_fn,
    )
    _register(net, hook, modules, match_names, True)
    return net


def backward_hook(
    net,
    modules=None,
    match_names=None,
    do_grad_input=False,
    do_grad_output=True,
    tag='',
    save_path='.',
    replace=True,
    histogram=True,
    bins=100,
    rename_fn=None,
):
    """Registers a backward hook to a network's modules for vizualization of the gradients.

    When net.backward() is called, the hook saves an image grid or a histogram 
    of grad_input/grad_output of the specified modules.

    Args:
        net (nn.Module): The network whose gradients are to be visualized.
        modules (list or tuple, optional): List of class definitions for the
            modules where the hook is attached e.g. nn.Conv2d  (default None).
        match_names (list or tuple, optional): List of strings. If any modules
            contain one of the strings then the hook is attached (default None).
        do_grad_input (bool, optional): If True the grad_input of the module is 
            visualized (default False).
        do_grad_output (bool, optional): If True the grad_output of the module 
            is visualized (default True).
        tag (str, optional): String tag to attach to saved images (default None).
        save_path (str, optional): Path to save visualisation results 
            (default '.').
        replace (bool, optional): If True, the images (from the same module) 
            are replaced whenever the hook is called (default True).
        histogram (bool, optional): If True then the visualization is a
            histrogram, otherwise it's an image grid.
        bins (bool, optional): Number of bins for histogram, if `histogram` is
            True (default 100).
    
    Note:
        * If modules or match_names are not provided then no hooks will be
          attached.
    """
    hook = _hook_generator(
        do_grad_input,
        do_grad_output,
        tag,
        save_path,
        replace,
        histogram,
        bins,
        'backward',
        rename_fn=rename_fn,
    )
    _register(net, hook, modules, match_names, False)
    return net


def parameters_hook(
    net,
    modules=None,
    match_names=None,
    param_names=None,
    tag='',
    save_path='.',
    replace=True,
    histogram=True,
    bins=100,
    rename_fn=None,
):
    """Registers a forward hook to a network's modules for vizualization of its parameters.

    When net.forward() is called, the hook saves an image grid or a histogram 
    of the parameters of the specified modules.

    Args:
        net (nn.Module): The network whose parameters are to be visualized.
        modules (list or tuple, optional): List of class definitions for the
            modules where the hook is attached e.g. nn.Conv2d  (default None).
        match_names (list or tuple, optional): List of strings. If any modules
            contain one of the strings then the hook is attached (default None).
        param_names (list or tuple, optional): List of strings. If any
            parameters of the module contain one of the strings then they are
            visualized (default None).
        tag (str, optional): String tag to attach to saved images (default None).
        save_path (str, optional): Path to save visualisation results 
            (default '.').
        replace (bool, optional): If True, the images (from the same module) 
            are replaced whenever the hook is called (default True).
        histogram (bool, optional): If True then the visualization is a
            histrogram, otherwise it's an image grid.
        bins (bool, optional): Number of bins for histogram, if `histogram` is
            True (default 100).

    Note:
        * If modules or match_names are not provided then no hooks will be
          attached.
        * If param_names are not provided then no parameters will be visualized.
    """
    hook = _hook_generator(
        False,
        False,
        tag,
        save_path,
        replace,
        histogram,
        bins,
        'parameters',
        param_names,
        rename_fn=rename_fn,
    )
    _register(net, hook, modules, match_names, True)
    return net


def visualize_parameters(
    net,
    modules=None,
    match_names=None,
    param_names=None,
    tag='',
    save_path='.',
    histogram=True,
    bins=100,
    rename_fn=None,
):
    """Visualizes a network's parameters on an image grid or histogram.

    Args:
        net (nn.Module): The network whose parameters are to be visualized.
        modules (list or tuple, optional): List of class definitions for the
            modules where the hook is attached e.g. nn.Conv2d  (default None).
        match_names (list or tuple, optional): List of strings. If any modules
            contain one of the strings then the hook is attached (default None).
        param_names (list or tuple, optional): List of strings. If any
            parameters of the module contain one of the strings then they are
            visualized (default None).
        tag (str, optional): String tag to attach to saved images (default None).
        save_path (str, optional): Path to save visualisation results 
            (default '.').
        histogram (bool, optional): If True then the visualization is a
            histrogram, otherwise it's an image grid.
        bins (bool, optional): Number of bins for histogram, if `histogram` is
            True (default 100).

    Note:
        * If modules or match_names are not provided then no parameters will be
          visualized.
        * If param_names are not provided then no parameters will be visualized.
    """
    save_path = bd.process_path(save_path, True)
    modules = process_none(modules)
    match_names = process_none(match_names)
    rename_fn = rename_fn or bd.identity
    for module_name, mod in net.named_modules():
        name_match = any(
            [torch.typename(modules).find(x) >= 0 for x in match_names]
        )
        instance_match = any([isinstance(mod, x) for x in modules])
        if instance_match or name_match:
            params = {x: _get_tensor(getattr(mod, x)) for x in param_names}
            for tensor_name, data in params.items():
                title = '{0}-{1}-{2}'.format(
                    tag, rename_fn(module_name), tensor_name
                )
                if data is None:
                    continue
                if histogram:
                    img = bd.torch2cv(data)
                    df = pd.DataFrame(img.reshape(img.size))
                    fig, ax = plt.subplots()
                    df.hist(bins=bins, ax=ax)
                    fig.savefig(
                        os.path.join(save_path, '{0}.png'.format(title))
                    )
                    plt.close(fig)
                else:
                    if data.dim() > 1:
                        img = bd.torch2cv(bd.make_grid(data))
                        to_save = (bd.map_range(img) * 255).astype(int)
                        cv2.imwrite(
                            os.path.join(save_path, '{0}.png'.format(title)),
                            to_save,
                        )
