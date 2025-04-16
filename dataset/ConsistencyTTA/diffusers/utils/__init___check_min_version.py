def check_min_version(min_version):
    if version.parse(__version__) < version.parse(min_version):
        if 'dev' in min_version:
            error_message = (
                'This example requires a source install from HuggingFace diffusers (see `https://huggingface.co/docs/diffusers/installation#install-from-source`),'
                )
        else:
            error_message = (
                f'This example requires a minimum version of {min_version},')
        error_message += f' but the version found is {__version__}.\n'
        raise ImportError(error_message)
