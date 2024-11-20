def tearDown(self):
    for path in self.teardown_tmp_dirs:
        shutil.rmtree(path, ignore_errors=True)
    self.teardown_tmp_dirs = []
