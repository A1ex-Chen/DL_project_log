def main():
    from threading import Lock
    assert torch.cuda.is_available() and torch.cuda.device_count(
        ) >= 2, 'This example is expected to run with at least 2 CUDA devices.'
    lock = Lock()
    device_ids = [0, 1]
    device = torch.device('cuda', device_ids[0])

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
        inputs = torch.tensor([1, 2, 3, 4]), [torch.tensor([10, 20, 30, 40]
            ), torch.tensor([100, 200, 300, 400])]
        print('inputs =', inputs)
        outputs = model.forward(*inputs)
        print('outputs =', outputs)
    data_parallel_case()

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
        inputs = Bunch([torch.tensor(1), torch.tensor([2, 2]), torch.tensor
            ([[3, 3], [4, 4]])]), [Bunch([torch.tensor(10), torch.tensor(20
            ), torch.tensor(30), torch.tensor(40)]), (Bunch([torch.tensor(
            100), torch.tensor(200), torch.tensor(300), torch.tensor(400)]),
            Bunch([torch.tensor(1000), torch.tensor(2000), torch.tensor(
            3000), torch.tensor(4000)]))]
        print('inputs =', inputs)
        outputs = model.forward(*inputs)
        print('outputs =', outputs)
    bunch_data_parallel_case_1()

    def bunch_data_parallel_case_2():


        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, input1: Bunch):
                with lock:
                    print('input1 =', input1)
                return input1
        print('===== BunchDataParallel (2) =====')
        model = Model()
        model = BunchDataParallel(model, device_ids).to(device)
        input1 = Bunch([torch.tensor(1), torch.tensor([2, 2]), torch.tensor
            ([[3, 3], [4, 4]])])
        print('input1 =', input1)
        output = model.forward(input1)
        print('output =', output)
    bunch_data_parallel_case_2()

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
    bunch_data_parallel_case_3()
