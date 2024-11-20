def __init__(self, *args, **kwargs):
    super(FixLanguageEncoder, self).__init__(*args, **kwargs)
    self.logit_scale = nn.Parameter(torch.ones([]), requires_grad=False)
