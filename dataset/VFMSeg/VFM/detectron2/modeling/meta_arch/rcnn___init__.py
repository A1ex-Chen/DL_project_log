@configurable
def __init__(self, *, backbone: Backbone, proposal_generator: nn.Module,
    pixel_mean: Tuple[float], pixel_std: Tuple[float]):
    """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
    super().__init__()
    self.backbone = backbone
    self.proposal_generator = proposal_generator
    self.register_buffer('pixel_mean', torch.tensor(pixel_mean).view(-1, 1,
        1), False)
    self.register_buffer('pixel_std', torch.tensor(pixel_std).view(-1, 1, 1
        ), False)
