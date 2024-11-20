@register_to_config
def __init__(self, embedding_dim: int=768):
    super().__init__()
    self.mean = nn.Parameter(torch.zeros(1, embedding_dim))
    self.std = nn.Parameter(torch.ones(1, embedding_dim))
