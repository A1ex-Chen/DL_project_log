def _pad_1x1_to_3x3_tensor(self, kernel1x1):
    if kernel1x1 is None:
        return 0
    else:
        return nn.functional.pad(kernel1x1, [1, 1, 1, 1])
