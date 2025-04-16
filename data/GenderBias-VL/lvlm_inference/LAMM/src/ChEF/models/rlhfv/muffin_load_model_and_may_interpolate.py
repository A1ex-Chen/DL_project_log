def load_model_and_may_interpolate(checkpoint_model, model):
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k
            ].shape:
            print(f'Removing key {k} from pretrained checkpoint')
            del checkpoint_model[k]
    for pos_embed_key in ('vision_pos_embed', 'pos_embed',
        'beit3.encoder.embed_positions.A.weight'):
        if pos_embed_key in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model[pos_embed_key]
            embedding_size = pos_embed_checkpoint.shape[-1]
            if pos_embed_key == 'beit3.encoder.embed_positions.A.weight':
                torchscale_model = True
                num_patches = model.beit3.vision_embed.num_patches
                num_extra_tokens = (model.beit3.vision_embed.
                    num_position_embeddings() + 2 - num_patches)
            else:
                torchscale_model = False
                num_patches = model.patch_embed.num_patches
                num_extra_tokens = getattr(model, pos_embed_key).shape[-2
                    ] - num_patches
            orig_size = int((pos_embed_checkpoint.shape[-2] -
                num_extra_tokens) ** 0.5)
            new_size = int(num_patches ** 0.5)
            if orig_size != new_size:
                print('Position interpolate from %dx%d to %dx%d' % (
                    orig_size, orig_size, new_size, new_size))
                if torchscale_model:
                    extra_tokens = pos_embed_checkpoint[:num_extra_tokens
                        ].unsqueeze(0)
                    pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
                else:
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                    embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(pos_tokens,
                    size=(new_size, new_size), mode='bicubic',
                    align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                if torchscale_model:
                    new_pos_embed = new_pos_embed.squeeze(0)
                checkpoint_model[pos_embed_key] = new_pos_embed
    return checkpoint_model
