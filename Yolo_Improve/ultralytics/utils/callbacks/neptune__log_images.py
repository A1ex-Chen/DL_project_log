def _log_images(imgs_dict, group=''):
    """Log scalars to the NeptuneAI experiment logger."""
    if run:
        for k, v in imgs_dict.items():
            run[f'{group}/{k}'].upload(File(v))
