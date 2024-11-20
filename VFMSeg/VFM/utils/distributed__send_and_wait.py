def _send_and_wait(r):
    if rank == r:
        tensor = torch.tensor(0, device='cuda')
    else:
        tensor = torch.tensor(1, device='cuda')
    dist.broadcast(tensor, r)
    while tensor.item() == 1:
        time.sleep(1)
