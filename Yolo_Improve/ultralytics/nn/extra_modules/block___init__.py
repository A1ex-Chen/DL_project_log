def __init__(self, inc_list, fusion='bifpn') ->None:
    super().__init__()
    assert fusion in ['weight', 'adaptive', 'concat', 'bifpn', 'SDI']
    self.fusion = fusion
    if self.fusion == 'bifpn':
        self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=
            torch.float32), requires_grad=True)
        self.relu = nn.ReLU()
        self.epsilon = 0.0001
    elif self.fusion == 'SDI':
        self.SDI = SDI(inc_list)
    else:
        self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in
            inc_list])
        if self.fusion == 'adaptive':
            self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)
