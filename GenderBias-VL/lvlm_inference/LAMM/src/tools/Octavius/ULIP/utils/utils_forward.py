@staticmethod
def forward(ctx, x):
    output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(output, x)
    return tuple(output)
