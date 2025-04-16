def bunch_data_parallel_case_3():


    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, input1: Bunch):
            with lock:
                print('input1 =', input1)
            return input1
    print('===== BunchDataParallel (3) =====')
    model = Model()
    model = BunchDataParallel(model, device_ids).to(device)
    input1 = Bunch([torch.tensor(1)])
    print('input1 =', input1)
    output = model.forward(input1)
    print('output =', output)
