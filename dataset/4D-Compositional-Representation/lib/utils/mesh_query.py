def query(self, size):
    active_simplices = self.active_simplices()
    active_simplices_points = self.points[active_simplices]
    new_points = sample_tetraheda(active_simplices_points, size=size)
    return new_points
