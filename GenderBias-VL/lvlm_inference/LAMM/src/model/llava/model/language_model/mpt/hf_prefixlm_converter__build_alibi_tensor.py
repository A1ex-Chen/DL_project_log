def _build_alibi_tensor(self: BloomModel, batch_size: int, query_length:
    int, key_length: int, dtype: torch.dtype, device: torch.device
    ) ->torch.Tensor:
    num_heads = self.config.n_head
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(2 ** -2 ** -(math.log2(closest_power_of_2) - 3),
        device=device, dtype=torch.float32)
    powers = torch.arange(1, 1 + closest_power_of_2, device=device, dtype=
        torch.int32)
    slopes = torch.pow(base, powers)
    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(2 ** -2 ** -(math.log2(2 *
            closest_power_of_2) - 3), device=device, dtype=torch.float32)
        num_remaining_heads = min(closest_power_of_2, num_heads -
            closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2,
            device=device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0
            )
    qa = torch.arange(query_length, device=device, dtype=torch.int32).view(
        -1, 1)
    ka = torch.arange(key_length, device=device, dtype=torch.int32).view(1, -1)
    diffs = qa - ka + key_length - query_length
    diffs = -diffs.abs()
    alibi = slopes.view(1, num_heads, 1, 1) * diffs.view(1, 1, query_length,
        key_length)
    alibi = alibi.expand(batch_size, -1, -1, -1).reshape(-1, query_length,
        key_length)
    return alibi.to(dtype)
