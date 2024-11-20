def interpolate_pos_embed(model, state_dict, interpolation: str='bicubic',
    seq_dim=1):
    old_pos_embed = state_dict.get('positional_embedding', None)
    grid_size = round((model.positional_embedding.shape[0] - 1) ** 0.5)
    if old_pos_embed is None:
        return
    grid_size = to_2tuple(grid_size)
    extra_tokens = 1
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return
    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[
            extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))
    print('Resizing position embedding grid-size from %s to %s',
        old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1
        ).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(pos_emb_img, size=grid_size, mode=
        interpolation, align_corners=True)
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] *
        grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['positional_embedding'] = new_pos_embed
