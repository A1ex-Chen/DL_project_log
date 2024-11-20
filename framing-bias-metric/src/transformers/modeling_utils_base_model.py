@property
def base_model(self) ->nn.Module:
    """
        :obj:`torch.nn.Module`: The main body of the model.
        """
    return getattr(self, self.base_model_prefix, self)
