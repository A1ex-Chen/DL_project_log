def broadcast_data(data):
    if not torch.distributed.is_initialized():
        return data
    rank = dist.get_rank()
    if rank == 0:
        data_tensor = torch.tensor(data + [0], device='cuda')
    else:
        data_tensor = torch.tensor(data + [1], device='cuda')
    torch.distributed.broadcast(data_tensor, 0)
    while data_tensor.cpu().numpy()[-1] == 1:
        time.sleep(1)
    return data_tensor.cpu().numpy().tolist()[:-1]
