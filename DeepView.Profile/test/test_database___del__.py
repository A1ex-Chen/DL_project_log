def __del__(self):
    if os.path.exists('test.sqlite'):
        os.remove('test.sqlite')
