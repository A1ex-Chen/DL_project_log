@unittest.skipIf(disabled, 'amp_C is unavailable')
def test_fuzz(self):
    input_size_pairs = (7777 * 77, 555 * 555), (777, 555), (555, 2048 * 32 + 1
        ), (2048 * 32 + 1, 555), (555, 2048 * 32), (2048 * 32, 555), (33333,
        555), (555, 33333)
    appliers = MultiTensorApply(2048 * 32), MultiTensorApply(333
        ), MultiTensorApply(33333)
    repeat_tensors = 1, 55
    for sizea, sizeb in input_size_pairs:
        for applier in appliers:
            for repeat in repeat_tensors:
                for in_type in (torch.float32, torch.float16):
                    for out_type in (torch.float32, torch.float16):
                        for inplace in (True, False):
                            if inplace is True and out_type is not in_type:
                                continue
                            else:
                                self.downscale(sizea, sizeb, applier,
                                    repeat, in_type, out_type, inplace=inplace)
                                self.find_inf(sizea, sizeb, applier, repeat,
                                    in_type, out_type, 0, 0, float('nan'),
                                    inplace=inplace)
                                self.find_inf(sizea, sizeb, applier, repeat,
                                    in_type, out_type, 2 * repeat - 1, 
                                    sizeb - 1, float('inf'), inplace=inplace)
                                self.find_inf(sizea, sizeb, applier, repeat,
                                    in_type, out_type, 2 * (repeat // 2), 
                                    sizea // 2, float('inf'), inplace=inplace)
