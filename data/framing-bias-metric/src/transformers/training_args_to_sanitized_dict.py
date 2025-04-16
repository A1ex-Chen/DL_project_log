def to_sanitized_dict(self) ->Dict[str, Any]:
    """
        Sanitized serialization to use with TensorBoard’s hparams
        """
    d = self.to_dict()
    d = {**d, **{'train_batch_size': self.train_batch_size,
        'eval_batch_size': self.eval_batch_size}}
    valid_types = [bool, int, float, str]
    if is_torch_available():
        valid_types.append(torch.Tensor)
    return {k: (v if type(v) in valid_types else str(v)) for k, v in d.items()}
