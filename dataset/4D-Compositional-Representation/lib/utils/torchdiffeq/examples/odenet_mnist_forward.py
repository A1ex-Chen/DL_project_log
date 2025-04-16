def forward(self, x):
    shape = torch.prod(torch.tensor(x.shape[1:])).item()
    return x.view(-1, shape)
