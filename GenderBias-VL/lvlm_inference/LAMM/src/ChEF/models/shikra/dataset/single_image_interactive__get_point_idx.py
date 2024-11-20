def _get_point_idx(self, point):
    assert isinstance(point, (tuple, list))
    assert isinstance(point[0], (int, float))
    assert len(point) == 2
    point = tuple(point)
    if point not in self.points:
        self.points.append(tuple(point))
        return len(self.points) - 1
    else:
        return self.points.index(point)
