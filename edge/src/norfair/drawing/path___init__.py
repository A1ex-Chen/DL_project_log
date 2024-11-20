def __init__(self, get_points_to_draw: Optional[Callable[[np.array], np.
    array]]=None, thickness: Optional[int]=None, color: Optional[Tuple[int,
    int, int]]=None, radius: Optional[int]=None, max_history=20):
    if get_points_to_draw is None:

        def get_points_to_draw(points):
            return [np.mean(np.array(points), axis=0)]
    self.get_points_to_draw = get_points_to_draw
    self.radius = radius
    self.thickness = thickness
    self.color = color
    self.past_points = defaultdict(lambda : [])
    self.max_history = max_history
    self.alphas = np.linspace(0.99, 0.01, max_history)
