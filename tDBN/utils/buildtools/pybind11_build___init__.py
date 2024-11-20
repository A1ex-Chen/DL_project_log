def __init__(self, sources, target, std='c++11', includes: list=None,
    defines: dict=None, cflags: str=None, libraries: dict=None, lflags: str
    =None, extra_cflags: str=None, extra_lflags: str=None, build_directory:
    str=None):
    pb11_includes = subprocess.check_output('python3 -m pybind11 --includes',
        shell=True).decode('utf8').strip('\n')
    cflags = cflags or '-fPIC -O3 '
    cflags += pb11_includes
    super().__init__(sources, target, std, includes, defines, cflags,
        libraries=libraries, lflags=lflags, extra_cflags=extra_cflags,
        extra_lflags=extra_lflags, build_directory=build_directory)
