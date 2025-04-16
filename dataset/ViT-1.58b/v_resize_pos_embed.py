def resize_pos_embed(posemb: torch.Tensor, posemb_new: torch.Tensor,
    num_prefix_tokens: int=1, gs_new: Tuple[int, int]=(), interpolation:
    str='bicubic', antialias: bool=False) ->torch.Tensor:
    """ Rescale the grid of position embeddings when loading from state_dict.

    *DEPRECATED* This function is being deprecated in favour of resample_abs_pos_embed

    Adapted from:
        https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[
            0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info(
        f'Resized position embedding: {posemb.shape} ({[gs_old, gs_old]}) to {posemb_new.shape} ({gs_new}).'
        )
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2
        )
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=
        interpolation, antialias=antialias, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] *
        gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb
