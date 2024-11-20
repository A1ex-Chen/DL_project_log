def __init__(self, obj: Union[Detection, TrackedObject]=None, points: np.
    ndarray=None, id: Any=None, label: Any=None, scores: np.ndarray=None,
    live_points: np.ndarray=None) ->None:
    if isinstance(obj, Detection):
        self.points = obj.points
        self.id = None
        self.label = obj.label
        self.scores = obj.scores
        self.live_points = np.ones(obj.points.shape[0]).astype(bool)
    elif isinstance(obj, TrackedObject):
        self.points = obj.estimate
        self.id = obj.id
        self.label = obj.label
        self.scores = None
        self.live_points = obj.live_points
    elif obj is None:
        self.points = points
        self.id = id
        self.label = label
        self.scores = scores
        self.live_points = live_points
    else:
        raise ValueError(
            f'Extecting a Detection or a TrackedObject but received {type(obj)}'
            )
