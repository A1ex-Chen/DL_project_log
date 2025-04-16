@property
def dtype(self) ->torch.dtype:
    """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
    return get_parameter_dtype(self)
