def load_language_adapter(self, model_path: str, num_token: int, dim: int,
    dim_out: int, tensor_norm: torch.Tensor, mult: int=2, depth: int=5):
    device = self._execution_device
    self.tensor_norm = tensor_norm.to(device)
    self.language_adapter = TranslatorNoLN(num_tok=num_token, dim=dim,
        dim_out=dim_out, mult=mult, depth=depth).to(device)
    self.language_adapter.load_state_dict(torch.load(model_path))
