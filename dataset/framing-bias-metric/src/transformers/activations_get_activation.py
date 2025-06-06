def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError('function {} not found in ACT2FN mapping {}'.format(
            activation_string, list(ACT2FN.keys())))
