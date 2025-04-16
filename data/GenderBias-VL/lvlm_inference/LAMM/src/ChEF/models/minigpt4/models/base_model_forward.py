@staticmethod
def forward(ctx, x):
    output = [torch.zeros_like(x) for _ in range(torch.distributed.
        get_world_size())]
    torch.distributed.all_gather(output, x)
    return tuple(output)
