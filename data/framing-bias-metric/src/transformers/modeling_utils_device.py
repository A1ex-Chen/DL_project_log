@property
def device(self) ->device:
    """
        :obj:`torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
    try:
        return next(self.parameters()).device
    except StopIteration:

        def find_tensor_attributes(module: nn.Module) ->List[Tuple[str, Tensor]
            ]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.
                is_tensor(v)]
            return tuples
        gen = self._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device
