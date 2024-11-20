def __init__(self, controlnets: Union[List[ControlNetModel], Tuple[
    ControlNetModel]]):
    super().__init__()
    self.nets = nn.ModuleList(controlnets)
