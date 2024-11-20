def find_anaconda():
    path = Path.home() / 'anaconda3'
    if path.exists():
        return path
    try:
        info = subprocess.check_output('conda info', shell=True).decode('utf-8'
            )
        info_dict = _get_info_from_anaconda_info(info)
        return info_dict['activeenvlocation']
    except subprocess.CalledProcessError:
        raise RuntimeError('find anadonda failed')
