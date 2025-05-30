def __init__(self, shape, n_class, channel, kernel_size, n_block,
    n_res_block, res_channel, attention=True, dropout=0.1, n_cond_res_block
    =0, cond_res_channel=0, cond_res_kernel=3, n_out_res_block=0):
    super().__init__()
    height, width = shape
    self.n_class = n_class
    if kernel_size % 2 == 0:
        kernel = kernel_size + 1
    else:
        kernel = kernel_size
    self.horizontal = CausalConv2d(n_class, channel, [kernel // 2, kernel],
        padding='down')
    self.vertical = CausalConv2d(n_class, channel, [(kernel + 1) // 2, 
        kernel // 2], padding='downright')
    coord_x = (torch.arange(height).float() - height / 2) / height
    coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
    coord_y = (torch.arange(width).float() - width / 2) / width
    coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
    self.register_buffer('background', torch.cat([coord_x, coord_y], 1))
    self.blocks = nn.ModuleList()
    for i in range(n_block):
        self.blocks.append(PixelBlock(channel, res_channel, kernel_size,
            n_res_block, attention=attention, dropout=dropout,
            condition_dim=cond_res_channel))
    if n_cond_res_block > 0:
        self.cond_resnet = CondResNet(n_class, cond_res_channel,
            cond_res_kernel, n_cond_res_block)
    out = []
    for i in range(n_out_res_block):
        out.append(GatedResBlock(channel, res_channel, 1))
    out.extend([nn.ELU(inplace=True), WNConv2d(channel, n_class, 1)])
    self.out = nn.Sequential(*out)
