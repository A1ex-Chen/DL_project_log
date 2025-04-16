@staticmethod
def set_weights(params, new_params):
    """
        Copies parameters from new_params to params

        :param params: dst parameters
        :param new_params: src parameters
        """
    for param, new_param in zip(params, new_params):
        param.data.copy_(new_param.data)
