@register_to_config
def __init__(self, hidden_size: int=768):
    super().__init__()
    self.hidden_size = hidden_size
    self.project = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
