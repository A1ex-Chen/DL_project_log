def get_env(self):
    """
        Return a copy of the ``os.environ`` object that sets up ``PYTHONPATH`` correctly, depending on the test suite
        it's invoked from. This is useful for invoking external programs from the test suite - e.g. distributed
        training.

        It always inserts ``./src`` first, then ``./tests`` or ``./examples`` depending on the test suite type and
        finally the preset ``PYTHONPATH`` if any (all full resolved paths).

        """
    env = os.environ.copy()
    paths = [self.src_dir_str]
    if '/examples' in self.test_file_dir_str:
        paths.append(self.examples_dir_str)
    else:
        paths.append(self.tests_dir_str)
    paths.append(env.get('PYTHONPATH', ''))
    env['PYTHONPATH'] = ':'.join(paths)
    return env
