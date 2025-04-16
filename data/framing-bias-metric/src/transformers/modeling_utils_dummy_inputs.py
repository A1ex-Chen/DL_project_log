@property
def dummy_inputs(self) ->Dict[str, torch.Tensor]:
    """
        :obj:`Dict[str, torch.Tensor]`: Dummy inputs to do a forward pass in the network.
        """
    return {'input_ids': torch.tensor(DUMMY_INPUTS)}
