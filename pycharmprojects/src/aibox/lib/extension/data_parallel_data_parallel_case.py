def data_parallel_case():


    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x: Tensor, x_list: List[Tensor]):
            with lock:
                print('x =', x)
                print('x_list =', x_list)
            return x, x_list
    num_devices = torch.cuda.device_count()
    print('num_devices:', num_devices)
    print('===== DataParallel =====')
    model = Model()
    model = DataParallel(model, device_ids).to(device)
    inputs = torch.tensor([1, 2, 3, 4]), [torch.tensor([10, 20, 30, 40]),
        torch.tensor([100, 200, 300, 400])]
    print('inputs =', inputs)
    outputs = model.forward(*inputs)
    print('outputs =', outputs)
