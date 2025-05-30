def all_gather(data):
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to('cuda')
    local_size = torch.IntTensor([tensor.numel()]).to('cuda')
    size_list = [torch.IntTensor([1]).to('cuda') for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to('cuda'))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to('cuda')
        tensor = torch.cat((tensor, padding), 0)
    dist.all_gather(tensor_list, tensor)
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list
