def __init__(self, num_classes=80, num_layers=3, inplace=True, head_layers=None
    ):
    super().__init__()
    assert head_layers is not None
    self.nc = num_classes
    self.no = num_classes + 5
    self.nl = num_layers
    self.grid = [torch.zeros(1)] * num_layers
    self.prior_prob = 0.01
    self.inplace = inplace
    stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64]
    self.stride = torch.tensor(stride)
    self.grid_cell_offset = 0.5
    self.grid_cell_size = 5.0
    self.stems = nn.ModuleList()
    self.cls_convs = nn.ModuleList()
    self.reg_convs = nn.ModuleList()
    self.cls_preds = nn.ModuleList()
    self.reg_preds = nn.ModuleList()
    for i in range(num_layers):
        idx = i * 5
        self.stems.append(head_layers[idx])
        self.cls_convs.append(head_layers[idx + 1])
        self.reg_convs.append(head_layers[idx + 2])
        self.cls_preds.append(head_layers[idx + 3])
        self.reg_preds.append(head_layers[idx + 4])
