def __setstate__(self, state):
    raise RuntimeError(
        'FP16_Optimizer should be deserialized using load_state_dict().')
