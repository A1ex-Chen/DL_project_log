def forward(self, i=None):
    if self.fx_only:
        if self.order == 2:
            fxfy = torch.stack([self.fx ** 2, self.fx ** 2])
        else:
            fxfy = torch.stack([self.fx, self.fx])
    elif self.order == 2:
        fxfy = torch.stack([self.fx ** 2, self.fy ** 2])
    else:
        fxfy = torch.stack([self.fx, self.fy])
    return fxfy
