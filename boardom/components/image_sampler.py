import os
import boardom as bd
from .util import _create_state_dict_element, _prepare_cfg
from ..config.common import IMAGE_SAMPLER_KEYS


# save_fn is save(obj, filename)
class ImageSampler(bd.Engine):
    def save_sample(
        self,
        sample,
        name='sample',
        directory='image_samples',
        overwrite=False,
        use_timestamps=True,
        save_fn=bd.imwrite,
        extension='.jpg',
        tag='',
        force_no_overwrite=False,
    ):
        with bd.interrupt_guard(reason='Saving image samples'):
            _save(
                sample,
                name=name,
                directory=directory,
                overwrite=overwrite,
                use_timestamps=use_timestamps,
                save_fn=save_fn,
                extension=extension,
                tag=tag,
                force_no_overwrite=force_no_overwrite,
            )

    def save_sample_from_cfg(
        self,
        sample,
        cfg=None,
        name='sample',
        save_fn=bd.imwrite,
        tag='',
        force_no_overwrite=False,
    ):
        cfg = _prepare_cfg(cfg, IMAGE_SAMPLER_KEYS + ['session_path'])
        directory = os.path.join(cfg.dg.session_path, 'image_samples')
        self.save_sample(
            sample,
            name=name,
            directory=directory,
            overwrite=cfg.overwrite,
            use_timestamps=cfg.use_timestamps,
            save_fn=save_fn,
            extension=cfg.extension,
            tag=tag,
            force_no_overwrite=force_no_overwrite,
        )


def _save(
    sample,
    name,
    directory,
    overwrite,
    use_timestamps,
    save_fn,
    extension,
    tag,
    force_no_overwrite,
):
    bd.make_dir(directory)
    overwrite = overwrite and (not force_no_overwrite)
    timestamp = bd.timestamp()
    tag = f'.{str(tag)}' if tag else ''
    str_timestamp = f'.{timestamp}' if use_timestamps else ''
    forced_tag = '.f' if force_no_overwrite else ''
    basename = f'{name}{str_timestamp}{tag}{forced_tag}{extension}'
    # if not overwriting, make sure that we give a new name to the file
    # (incase it already exists)
    if not overwrite:
        _fullname = os.path.join(directory, basename)
        basename = os.path.basename(bd.number_file_if_exists(_fullname))
    full_name = os.path.join(directory, basename)
    safe_save(sample, full_name, save_fn)

    # If we are overwiting find and delete the previous_files
    # (EXCEPT THE FORCED ONES)
    if overwrite:
        to_remove = [
            x
            for x in os.listdir(directory)
            if x.startswith(name) and ('.forced.' not in x)
        ]
        for f in to_remove:
            os.remove(os.path.join(directory, f))


def safe_save(obj, full_name, save_func):
    if os.path.exists(full_name):
        # Write temporary new file
        tmp_file = full_name + '.tmp'
        save_func(obj, tmp_file)
        # Backup current file
        bak_current_file = full_name + '.bak'
        os.replace(full_name, bak_current_file)
        # Rename temporary file
        os.replace(tmp_file, full_name)
        # Remove backup
        os.remove(bak_current_file)
    else:
        save_func(obj, full_name)
