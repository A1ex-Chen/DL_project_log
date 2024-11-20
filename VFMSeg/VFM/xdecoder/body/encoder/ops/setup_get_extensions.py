def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'src')
    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    source_cpu = glob.glob(os.path.join(extensions_dir, 'cpu', '*.cpp'))
    source_cuda = glob.glob(os.path.join(extensions_dir, 'cuda', '*.cu'))
    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {'cxx': []}
    define_macros = []
    if (os.environ.get('FORCE_CUDA') or torch.cuda.is_available()
        ) and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        extra_compile_args['nvcc'] = ['-DCUDA_HAS_FP16=1',
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__']
    elif CUDA_HOME is None:
        raise NotImplementedError(
            'CUDA_HOME is None. Please set environment variable CUDA_HOME.')
    else:
        raise NotImplementedError(
            'No CUDA runtime is found. Please set FORCE_CUDA=1 or test it by running torch.cuda.is_available().'
            )
    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [extension('MultiScaleDeformableAttention', sources,
        include_dirs=include_dirs, define_macros=define_macros,
        extra_compile_args=extra_compile_args)]
    return ext_modules
