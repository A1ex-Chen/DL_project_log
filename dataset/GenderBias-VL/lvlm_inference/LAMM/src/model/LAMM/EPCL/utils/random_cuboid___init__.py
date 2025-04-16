def __init__(self, min_points, aspect=0.8, min_crop=0.5, max_crop=1.0,
    box_filter_policy='center'):
    self.aspect = aspect
    self.min_crop = min_crop
    self.max_crop = max_crop
    self.min_points = min_points
    self.box_filter_policy = box_filter_policy
