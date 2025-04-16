def resize_pos(self):
    pos_embed_checkpoint = (self.vision_tower.vision_model.embeddings.
        position_embedding.weight)
    pos_embed_checkpoint = pos_embed_checkpoint.unsqueeze(0)
    orig_size = 24
    new_size = 35
    if pos_embed_checkpoint.shape[1] == new_size ** 2 + 1:
        self.is_resize_pos = True
    else:
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_extra_tokens = 1
        new_num = new_size ** 2 + num_extra_tokens
        print('Position interpolate from %dx%d to %dx%d' % (orig_size,
            orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
            embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(
            new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        new_pos_embed = new_pos_embed.squeeze(0)
        self.vision_tower.vision_model.embeddings.position_embedding = (torch
            .nn.Embedding(new_num, 1024))
        (self.vision_tower.vision_model.embeddings.position_embedding.weight
            ) = (torch.nn.Parameter(new_pos_embed.to(pos_embed_checkpoint.
            dtype)))
        self.vision_tower.vision_model.embeddings.position_ids = torch.arange(
            new_num).expand((1, -1))
        self.is_resize_pos = True
