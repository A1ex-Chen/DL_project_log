def __init__(self, max_points: int=200, min_distance: int=15, block_size:
    int=3, transformations_getter: TransformationGetter=None, draw_flow:
    bool=False, flow_color: Optional[Tuple[int, int, int]]=None,
    quality_level: float=0.01):
    self.max_points = max_points
    self.min_distance = min_distance
    self.block_size = block_size
    self.draw_flow = draw_flow
    if self.draw_flow and flow_color is None:
        flow_color = [0, 0, 100]
    self.flow_color = flow_color
    self.gray_prvs = None
    self.prev_pts = None
    if transformations_getter is None:
        transformations_getter = HomographyTransformationGetter()
    self.transformations_getter = transformations_getter
    self.prev_mask = None
    self.gray_next = None
    self.quality_level = quality_level
