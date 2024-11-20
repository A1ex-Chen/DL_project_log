def _log_debug_samples(files, title='Debug Samples') ->None:
    """
    Log files (images) as debug samples in the ClearML task.

    Args:
        files (list): A list of file paths in PosixPath format.
        title (str): A title that groups together images with the same values.
    """
    import re
    if (task := Task.current_task()):
        for f in files:
            if f.exists():
                it = re.search('_batch(\\d+)', f.name)
                iteration = int(it.groups()[0]) if it else 0
                task.get_logger().report_image(title=title, series=f.name.
                    replace(it.group(), ''), local_path=str(f), iteration=
                    iteration)
