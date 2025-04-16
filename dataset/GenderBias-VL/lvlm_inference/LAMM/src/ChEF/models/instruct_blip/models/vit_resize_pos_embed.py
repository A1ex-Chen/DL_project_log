def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.
        shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:
            ]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    print('Position embedding grid-size from %s to %s', [gs_old, gs_old],
        gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2
        )
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic',
        align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] *
        gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return
