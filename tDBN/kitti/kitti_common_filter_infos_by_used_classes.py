def filter_infos_by_used_classes(infos, used_classes):
    new_infos = []
    for info in infos:
        annos = info['annos']
        name_in_info = False
        for name in used_classes:
            if name in annos['name']:
                name_in_info = True
                break
        if name_in_info:
            new_infos.append(info)
    return new_infos
