from typing import Tuple, List, Dict

import torch
from torch import Tensor
from torch.nn import DataParallel


class Bunch(tuple):



class BunchDataParallel(DataParallel):




if __name__ == '__main__':

        try:
            scattered_inputs = scatter_map(inputs)
            scattered_kwargs = scatter_map(kwargs)

            if len(scattered_inputs) < len(scattered_kwargs):
                scattered_inputs.extend([() for _ in range(len(scattered_kwargs) - len(scattered_inputs))])
            elif len(scattered_kwargs) < len(scattered_inputs):
                scattered_kwargs.extend([{} for _ in range(len(scattered_inputs) - len(scattered_kwargs))])
        finally:
            scatter_map = None

        return scattered_inputs, scattered_kwargs

    def gather(self, outputs: List[Tuple], target_device: int, dim: int = 0):
        # NOTE:
        #       Original `DataParallel`
        #
        #           # cuda:0
        #           outputs = [
        #               # cuda:0
        #               (
        #                   torch.tensor([1, 2]),               # Modality 1
        #                   [
        #                       torch.tensor([10, 20]),         # Modality 2
        #                       torch.tensor([100, 200])        # Modality 3
        #                   ]
        #               ),
        #               # cuda:1
        #               (
        #                   torch.tensor([3, 4]),               # Modality 1
        #                   [
        #                       torch.tensor([30, 40]),         # Modality 2
        #                       torch.tensor([300, 400])        # Modality 3
        #                   ]
        #               )
        #           ]
        #
        #       is gathered to
        #
        #           # cuda:0
        #           gathered_outputs = (
        #               torch.tensor([1, 2, 3, 4]),                 # Modality 1
        #               [
        #                   torch.tensor([10, 20, 30, 40]),         # Modality 2
        #                   torch.tensor([100, 200, 300, 400])      # Modality 3
        #               ]
        #           )
        #
        #       By overwritten `gather` in `BunchDataParallel`
        #
        #           outputs = [
        #               # cuda:0
        #               (
        #                   # Modality 1
        #                   Bunch([
        #                       torch.tensor(1),
        #                       torch.tensor([2, 2])
        #                   ]),
        #                   [
        #                       # Modality 2
        #                       Bunch([
        #                           torch.tensor(10),
        #                           torch.tensor(20)
        #                       ]),
        #                       (
        #                           # Modality 3
        #                           Bunch([
        #                               torch.tensor(100),
        #                               torch.tensor(200)
        #                           ]),
        #                           # Modality 4
        #                           Bunch([
        #                               torch.tensor(1000),
        #                               torch.tensor(2000)
        #                           ]),
        #                       )
        #                   ]
        #               ),
        #               # cuda:1
        #               (
        #                   # Modality 1
        #                   Bunch([
        #                       torch.tensor([[3, 3],
        #                                     [4, 4]])
        #                   ]),
        #                   [
        #                       # Modality 2
        #                       Bunch([
        #                           torch.tensor(30),
        #                           torch.tensor(40)
        #                       ]),
        #                       (
        #                           # Modality 3
        #                           Bunch([
        #                               torch.tensor(300),
        #                               torch.tensor(400)
        #                           ]),
        #                           # Modality 4
        #                           Bunch([
        #                               torch.tensor(3000),
        #                               torch.tensor(4000)
        #                           ]),
        #                       )
        #                   ]
        #               )
        #           ]
        #
        #       is gathered to
        #
        #           # cuda:0
        #           gathered_outputs = (
        #               # Modality 1
        #               Bunch([
        #                   torch.tensor(1),
        #                   torch.tensor([2, 2]),
        #                   torch.tensor([[3, 3],
        #                                 [4, 4]])
        #               ]),
        #               [
        #                   # Modality 2
        #                   Bunch([
        #                       torch.tensor(10),
        #                       torch.tensor(20),
        #                       torch.tensor(30),
        #                       torch.tensor(40)
        #                   ]),
        #                   (
        #                       # Modality 3
        #                       Bunch([
        #                           torch.tensor(100),
        #                           torch.tensor(200),
        #                           torch.tensor(300),
        #                           torch.tensor(400)
        #                       ]),
        #                       # Modality 4
        #                       Bunch([
        #                           torch.tensor(1000),
        #                           torch.tensor(2000),
        #                           torch.tensor(3000),
        #                           torch.tensor(4000)
        #                       ])
        #                   )
        #               ]
        #           )


        try:
            gathred_outputs = gather_map(outputs)
        finally:
            gather_map = None

        return gathred_outputs


if __name__ == '__main__':
    def main():
        from threading import Lock

        assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, \
            'This example is expected to run with at least 2 CUDA devices.'

        lock = Lock()
        device_ids = [0, 1]
        device = torch.device('cuda', device_ids[0])

        # region ===== DataParallel Case =====

        data_parallel_case()
        # endregion ==========================

        # region ===== BunchDataParallel Case 1 =====

        bunch_data_parallel_case_1()
        # endregion ===============================

        # region ===== BunchDataParallel Case 2 =====

        bunch_data_parallel_case_2()
        # endregion ===============================

        # region ===== BunchDataParallel Case 3 =====


            num_devices = torch.cuda.device_count()
            print('num_devices:', num_devices)

            print('===== DataParallel =====')
            model = Model()
            model = DataParallel(model, device_ids).to(device)
            inputs = (
                torch.tensor([1, 2, 3, 4]),
                [
                    torch.tensor([10, 20, 30, 40]),
                    torch.tensor([100, 200, 300, 400])
                ]
            )
            print('inputs =', inputs)
            outputs = model.forward(*inputs)
            print('outputs =', outputs)

        data_parallel_case()
        # endregion ==========================

        # region ===== BunchDataParallel Case 1 =====
        def bunch_data_parallel_case_1():
            class Model(torch.nn.Module):


            print('===== BunchDataParallel (1) =====')
            model = Model()
            model = BunchDataParallel(model, device_ids).to(device)
            inputs = (
                Bunch([
                    torch.tensor(1),
                    torch.tensor([2, 2]),
                    torch.tensor([[3, 3],
                                  [4, 4]])
                ]),
                [
                    Bunch([
                        torch.tensor(10),
                        torch.tensor(20),
                        torch.tensor(30),
                        torch.tensor(40)
                    ]),
                    (
                        Bunch([
                            torch.tensor(100),
                            torch.tensor(200),
                            torch.tensor(300),
                            torch.tensor(400)
                        ]),
                        Bunch([
                            torch.tensor(1000),
                            torch.tensor(2000),
                            torch.tensor(3000),
                            torch.tensor(4000)
                        ])
                    )
                ]
            )
            print('inputs =', inputs)
            outputs = model.forward(*inputs)
            print('outputs =', outputs)

        bunch_data_parallel_case_1()
        # endregion ===============================

        # region ===== BunchDataParallel Case 2 =====
        def bunch_data_parallel_case_2():
            class Model(torch.nn.Module):


            print('===== BunchDataParallel (2) =====')
            model = Model()
            model = BunchDataParallel(model, device_ids).to(device)
            input1 = Bunch([
                torch.tensor(1),
                torch.tensor([2, 2]),
                torch.tensor([[3, 3],
                              [4, 4]])
            ])
            print('input1 =', input1)
            output = model.forward(input1)
            print('output =', output)

        bunch_data_parallel_case_2()
        # endregion ===============================

        # region ===== BunchDataParallel Case 3 =====
        def bunch_data_parallel_case_3():
            class Model(torch.nn.Module):


            print('===== BunchDataParallel (3) =====')
            model = Model()
            model = BunchDataParallel(model, device_ids).to(device)
            input1 = Bunch([
                torch.tensor(1)
            ])
            print('input1 =', input1)
            output = model.forward(input1)
            print('output =', output)

        bunch_data_parallel_case_3()
        # endregion ===============================

    main()