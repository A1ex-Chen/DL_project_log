@property
def dtype(self) ->dtype:
    """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
    try:
        return next(self.parameters()).dtype
    except StopIteration:

        def find_tensor_attributes(module: nn.Module) ->List[Tuple[str, Tensor]
            ]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.
                is_tensor(v)]
            return tuples
        gen = self._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype
