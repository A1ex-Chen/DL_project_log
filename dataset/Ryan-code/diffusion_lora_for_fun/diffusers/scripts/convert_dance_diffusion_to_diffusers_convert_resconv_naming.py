def convert_resconv_naming(name):
    if name.startswith('skip'):
        return name.replace('skip', RES_CONV_MAP['skip'])
    if not name.startswith('main.'):
        raise ValueError(f'ResConvBlock error with {name}')
    return name.replace(name[:6], RES_CONV_MAP[name[:6]])
