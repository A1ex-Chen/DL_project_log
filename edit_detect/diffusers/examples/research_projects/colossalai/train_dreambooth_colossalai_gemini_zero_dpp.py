def gemini_zero_dpp(model: torch.nn.Module, placememt_policy: str='auto'):
    from colossalai.nn.parallel import GeminiDDP
    model = GeminiDDP(model, device=get_current_device(), placement_policy=
        placememt_policy, pin_memory=True, search_range_mb=64)
    return model
