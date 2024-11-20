def forward_split_cat(self, x: Tensor) ->Tensor:
    x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
    x1 = self.partial_conv3(x1)
    x = torch.cat((x1, x2), 1)
    return x
