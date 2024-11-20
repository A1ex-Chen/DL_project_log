@staticmethod
def forward(x):
    return x * torch.sigmoid(x)
