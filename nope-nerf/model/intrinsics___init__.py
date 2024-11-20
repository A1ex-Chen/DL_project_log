def __init__(self, req_grad, fx_only, order=2, init_focal=None):
    super(LearnFocal, self).__init__()
    self.fx_only = fx_only
    self.order = order
    if self.fx_only:
        if init_focal is None:
            self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32),
                requires_grad=req_grad)
        else:
            if self.order == 2:
                coe_x = torch.tensor(np.sqrt(init_focal), requires_grad=False
                    ).float()
            elif self.order == 1:
                coe_x = torch.tensor(init_focal, requires_grad=False).float()
            else:
                print('Focal init order need to be 1 or 2. Exit')
                exit()
            self.fx = nn.Parameter(coe_x, requires_grad=req_grad)
    elif init_focal is None:
        self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32),
            requires_grad=req_grad)
        self.fy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32),
            requires_grad=req_grad)
    elif isinstance(init_focal, list):
        if self.order == 2:
            coe_x = torch.tensor(np.sqrt(init_focal[0]), requires_grad=False
                ).float()
            coe_y = torch.tensor(np.sqrt(init_focal[1]), requires_grad=False
                ).float()
        elif self.order == 1:
            coe_x = torch.tensor(init_focal[0], requires_grad=False).float()
            coe_y = torch.tensor(init_focal[1], requires_grad=False).float()
        else:
            print('Focal init order need to be 1 or 2. Exit')
            exit()
        self.fx = nn.Parameter(coe_x, requires_grad=req_grad)
        self.fy = nn.Parameter(coe_y, requires_grad=req_grad)
    else:
        if self.order == 2:
            coe_x = torch.tensor(np.sqrt(init_focal), requires_grad=False
                ).float()
            coe_y = torch.tensor(np.sqrt(init_focal), requires_grad=False
                ).float()
        elif self.order == 1:
            coe_x = torch.tensor(init_focal, requires_grad=False).float()
            coe_y = torch.tensor(init_focal, requires_grad=False).float()
        else:
            print('Focal init order need to be 1 or 2. Exit')
            exit()
        self.fx = nn.Parameter(coe_x, requires_grad=req_grad)
        self.fy = nn.Parameter(coe_y, requires_grad=req_grad)
