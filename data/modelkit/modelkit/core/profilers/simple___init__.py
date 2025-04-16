def __init__(self, model: Model) ->None:
    super().__init__(model)
    self.recording_hook: Dict[str, float] = {}
    self.durations = defaultdict(list)
    self.net_durations = defaultdict(list)
    graph: Dict[str, Set] = defaultdict(set)
    self.graph = self._build_graph(self.model, graph)
    self.graph_calls: Dict[str, Dict[str, int]] = defaultdict(lambda :
        defaultdict(int))
