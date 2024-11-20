def __getstate__(self):
    raise RuntimeError(
        'FP16_Optimizer should be serialized using state_dict().')
