def is_test(fname):
    if fname.startswith('tests'):
        return True
    if fname.startswith('examples') and fname.split(os.path.sep)[-1
        ].startswith('test'):
        return True
    return False
