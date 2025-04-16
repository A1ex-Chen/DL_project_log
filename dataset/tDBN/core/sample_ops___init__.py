def __init__(self, db_infos, groups, db_prepor=None, rate=1.0,
    global_rot_range=None):
    for k, v in db_infos.items():
        print(f'load {len(v)} {k} database infos')
    if db_prepor is not None:
        db_infos = db_prepor(db_infos)
        print('After filter database:')
        for k, v in db_infos.items():
            print(f'load {len(v)} {k} database infos')
    self.db_infos = db_infos
    self._rate = rate
    self._groups = groups
    self._group_db_infos = {}
    self._group_name_to_names = []
    self._sample_classes = []
    self._sample_max_nums = []
    self._use_group_sampling = False
    if any([(len(g) > 1) for g in groups]):
        self._use_group_sampling = True
    if not self._use_group_sampling:
        self._group_db_infos = self.db_infos
        for group_info in groups:
            group_names = list(group_info.keys())
            self._sample_classes += group_names
            self._sample_max_nums += list(group_info.values())
    else:
        for group_info in groups:
            group_dict = {}
            group_names = list(group_info.keys())
            group_name = ', '.join(group_names)
            self._sample_classes += group_names
            self._sample_max_nums += list(group_info.values())
            self._group_name_to_names.append((group_name, group_names))
            for name in group_names:
                for item in db_infos[name]:
                    gid = item['group_id']
                    if gid not in group_dict:
                        group_dict[gid] = [item]
                    else:
                        group_dict[gid] += [item]
            if group_name in self._group_db_infos:
                raise ValueError('group must be unique')
            group_data = list(group_dict.values())
            self._group_db_infos[group_name] = group_data
            info_dict = {}
            if len(group_info) > 1:
                for group in group_data:
                    names = [item['name'] for item in group]
                    names = sorted(names)
                    group_name = ', '.join(names)
                    if group_name in info_dict:
                        info_dict[group_name] += 1
                    else:
                        info_dict[group_name] = 1
            print(info_dict)
    self._sampler_dict = {}
    for k, v in self._group_db_infos.items():
        self._sampler_dict[k] = prep.BatchSampler(v, k)
    self._enable_global_rot = False
    if global_rot_range is not None:
        if not isinstance(global_rot_range, (list, tuple, np.ndarray)):
            global_rot_range = [-global_rot_range, global_rot_range]
        else:
            assert shape_mergeable(global_rot_range, [2])
        if np.abs(global_rot_range[0] - global_rot_range[1]) >= 0.001:
            self._enable_global_rot = True
    self._global_rot_range = global_rot_range
