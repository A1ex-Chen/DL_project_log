def setUp(self):
    self.teardown_tmp_dirs = []
    self._test_file_path = inspect.getfile(self.__class__)
    path = Path(self._test_file_path).resolve()
    self._test_file_dir = path.parents[0]
    for up in [1, 2, 3]:
        tmp_dir = path.parents[up]
        if (tmp_dir / 'src').is_dir() and (tmp_dir / 'tests').is_dir():
            break
    if tmp_dir:
        self._repo_root_dir = tmp_dir
    else:
        raise ValueError(
            f"can't figure out the root of the repo from {self._test_file_path}"
            )
    self._tests_dir = self._repo_root_dir / 'tests'
    self._examples_dir = self._repo_root_dir / 'examples'
    self._src_dir = self._repo_root_dir / 'src'
