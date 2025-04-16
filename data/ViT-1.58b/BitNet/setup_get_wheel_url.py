def get_wheel_url():
    torch_cuda_version = parse(torch.version.cuda)
    torch_version_raw = parse(torch.__version__)
    torch_cuda_version = parse('11.8'
        ) if torch_cuda_version.major == 11 else parse('12.2')
    python_version = f'cp{sys.version_info.major}{sys.version_info.minor}'
    platform_name = get_platform()
    mamba_ssm_version = get_package_version()
    cuda_version = f'{torch_cuda_version.major}{torch_cuda_version.minor}'
    torch_version = f'{torch_version_raw.major}.{torch_version_raw.minor}'
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    wheel_filename = (
        f'{PACKAGE_NAME}-{mamba_ssm_version}+cu{cuda_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl'
        )
    wheel_url = BASE_WHEEL_URL.format(tag_name=f'v{mamba_ssm_version}',
        wheel_name=wheel_filename)
    return wheel_url, wheel_filename
