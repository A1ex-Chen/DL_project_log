def get_useable_cuda():
    test_tensor = torch.zeros(1)
    useable_cuda = []
    for i in range(8):
        try:
            test_tensor.to(device=i)
            useable_cuda.append(i)
        except:
            continue
    return useable_cuda
