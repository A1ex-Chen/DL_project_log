def _get_info_from_anaconda_info(info, split=':'):
    info = info.strip('\n').replace(' ', '')
    info_dict = {}
    latest_key = ''
    for line in info.splitlines():
        if split in line:
            pair = line.split(split)
            info_dict[pair[0]] = pair[1]
            latest_key = pair[0]
        else:
            if not isinstance(info_dict[latest_key], list):
                info_dict[latest_key] = [info_dict[latest_key]]
            info_dict[latest_key].append(line)
    return info_dict
