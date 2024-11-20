def log_debug_samples(self, files, title='Debug Samples'):
    """
        Log files (images) as debug samples in the ClearML task.

        arguments:
        files (List(PosixPath)) a list of file paths in PosixPath format
        title (str) A title that groups together images with the same values
        """
    for f in files:
        if f.exists():
            it = re.search('_batch(\\d+)', f.name)
            iteration = int(it.groups()[0]) if it else 0
            self.task.get_logger().report_image(title=title, series=f.name.
                replace(it.group(), ''), local_path=str(f), iteration=iteration
                )
