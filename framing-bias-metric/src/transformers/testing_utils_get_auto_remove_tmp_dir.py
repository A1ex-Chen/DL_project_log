def get_auto_remove_tmp_dir(self, tmp_dir=None, before=None, after=None):
    """
        Args:
            tmp_dir (:obj:`string`, `optional`):
                if :obj:`None`:

                   - a unique temporary path will be created
                   - sets ``before=True`` if ``before`` is :obj:`None`
                   - sets ``after=True`` if ``after`` is :obj:`None`
                else:

                   - :obj:`tmp_dir` will be created
                   - sets ``before=True`` if ``before`` is :obj:`None`
                   - sets ``after=False`` if ``after`` is :obj:`None`
            before (:obj:`bool`, `optional`):
                If :obj:`True` and the :obj:`tmp_dir` already exists, make sure to empty it right away if :obj:`False`
                and the :obj:`tmp_dir` already exists, any existing files will remain there.
            after (:obj:`bool`, `optional`):
                If :obj:`True`, delete the :obj:`tmp_dir` at the end of the test if :obj:`False`, leave the
                :obj:`tmp_dir` and its contents intact at the end of the test.

        Returns:
            tmp_dir(:obj:`string`): either the same value as passed via `tmp_dir` or the path to the auto-selected tmp
            dir
        """
    if tmp_dir is not None:
        if before is None:
            before = True
        if after is None:
            after = False
        path = Path(tmp_dir).resolve()
        if not tmp_dir.startswith('./'):
            raise ValueError(
                f'`tmp_dir` can only be a relative path, i.e. `./some/path`, but received `{tmp_dir}`'
                )
        if before is True and path.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
    else:
        if before is None:
            before = True
        if after is None:
            after = True
        tmp_dir = tempfile.mkdtemp()
    if after is True:
        self.teardown_tmp_dirs.append(tmp_dir)
    return tmp_dir
