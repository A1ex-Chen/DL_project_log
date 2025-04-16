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
                for x_type in (torch.float32, torch.float16):
                    for y_type in (torch.float32, torch.float16):
                        for out_type in (torch.float32, torch.float16):
                            for inplace in (True, False):
                                if inplace is True and y_type is not out_type:
                                    continue
                                else:
                                    self.axpby(sizea, sizeb, applier,
                                        repeat, x_type, y_type, out_type,
                                        inplace=inplace)
