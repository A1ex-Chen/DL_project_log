def skip_unless(var, value):
    env = os.environ.get(var)
    return pytest.mark.skipif(env != value, reason=f'{var} is not {value}')
