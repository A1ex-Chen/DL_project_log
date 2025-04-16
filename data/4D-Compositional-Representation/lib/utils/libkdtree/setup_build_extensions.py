def build_extensions(self):
    comp = self.compiler.compiler_type
    if comp in ('unix', 'cygwin', 'mingw32'):
        if use_omp:
            extra_compile_args = ['-std=c99', '-O3', '-fopenmp']
            extra_link_args = ['-lgomp']
        else:
            extra_compile_args = ['-std=c99', '-O3']
            extra_link_args = []
    elif comp == 'msvc':
        extra_compile_args = ['/Ox']
        extra_link_args = []
        if use_omp:
            extra_compile_args.append('/openmp')
    else:
        raise ValueError(
            'Compiler flags undefined for %s. Please modify setup.py and add compiler flags'
             % comp)
    self.extensions[0].extra_compile_args = extra_compile_args
    self.extensions[0].extra_link_args = extra_link_args
    build_ext.build_extensions(self)
