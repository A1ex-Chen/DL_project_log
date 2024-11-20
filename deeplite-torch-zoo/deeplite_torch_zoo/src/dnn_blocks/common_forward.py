def forward(self, x):
    return x * torch.nn.functional.softplus(x).tanh()
