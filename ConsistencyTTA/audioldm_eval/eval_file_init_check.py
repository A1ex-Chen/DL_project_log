def file_init_check(self, dir):
    assert os.path.exists(dir), 'The path does not exist %s' % dir
    assert len(os.listdir(dir)) > 1, 'There is no files in %s' % dir
