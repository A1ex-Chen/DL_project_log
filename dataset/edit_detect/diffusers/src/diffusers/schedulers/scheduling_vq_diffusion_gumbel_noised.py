def gumbel_noised(logits: torch.Tensor, generator: Optional[torch.Generator]
    ) ->torch.Tensor:
    """
    Apply gumbel noise to `logits`
    """
    uniform = torch.rand(logits.shape, device=logits.device, generator=
        generator)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    noised = gumbel_noise + logits
    return noised
