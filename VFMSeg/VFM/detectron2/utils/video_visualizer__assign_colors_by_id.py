def _assign_colors_by_id(self, instances: Instances) ->List:
    colors = []
    untracked_ids = set(self._assigned_colors.keys())
    for id in instances.ID:
        if id in self._assigned_colors:
            colors.append(self._color_pool[self._assigned_colors[id]])
            untracked_ids.remove(id)
        else:
            assert len(self._color_idx_set
                ) >= 1, f'Number of id exceeded maximum,                     max = {self._max_num_instances}'
            idx = self._color_idx_set.pop()
            color = self._color_pool[idx]
            self._assigned_colors[id] = idx
            colors.append(color)
    for id in untracked_ids:
        self._color_idx_set.add(self._assigned_colors[id])
        del self._assigned_colors[id]
    return colors
