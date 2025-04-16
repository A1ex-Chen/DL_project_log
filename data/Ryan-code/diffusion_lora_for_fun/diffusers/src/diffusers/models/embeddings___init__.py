def __init__(self, IPAdapterImageProjectionLayers: Union[List[nn.Module],
    Tuple[nn.Module]]):
    super().__init__()
    self.image_projection_layers = nn.ModuleList(IPAdapterImageProjectionLayers
        )
