def _preprocess(self, db_infos):
    for name, min_num in self._min_gt_point_dict.items():
        if min_num > 0:
            filtered_infos = []
            for info in db_infos[name]:
                if info['num_points_in_gt'] >= min_num:
                    filtered_infos.append(info)
            db_infos[name] = filtered_infos
    return db_infos
