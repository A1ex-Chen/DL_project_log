def bunch_data_parallel_case_1():


    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, input1: Bunch, input2: List):
            with lock:
                print('input1 =', input1)
                print('input2 =', input2)
            return input1, input2
    print('===== BunchDataParallel (1) =====')
    model = Model()
    model = BunchDataParallel(model, device_ids).to(device)
    inputs = Bunch([torch.tensor(1), torch.tensor([2, 2]), torch.tensor([[3,
        3], [4, 4]])]), [Bunch([torch.tensor(10), torch.tensor(20), torch.
        tensor(30), torch.tensor(40)]), (Bunch([torch.tensor(100), torch.
        tensor(200), torch.tensor(300), torch.tensor(400)]), Bunch([torch.
        tensor(1000), torch.tensor(2000), torch.tensor(3000), torch.tensor(
        4000)]))]
    print('inputs =', inputs)
    outputs = model.forward(*inputs)
    print('outputs =', outputs)
