def get_dummy_token(self, batch_size: int, device: torch.device
    ) ->torch.Tensor:
    return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64,
        device=device)
