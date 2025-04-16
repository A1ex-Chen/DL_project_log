def convert_ip_adapter_image_proj_to_diffusers(self, state_dict):
    updated_state_dict = {}
    clip_embeddings_dim_in = state_dict['proj.0.weight'].shape[1]
    clip_embeddings_dim_out = state_dict['proj.0.weight'].shape[0]
    multiplier = clip_embeddings_dim_out // clip_embeddings_dim_in
    norm_layer = 'norm.weight'
    cross_attention_dim = state_dict[norm_layer].shape[0]
    num_tokens = state_dict['proj.2.weight'].shape[0] // cross_attention_dim
    image_projection = IPAdapterFullImageProjection(cross_attention_dim=
        cross_attention_dim, image_embed_dim=clip_embeddings_dim_in, mult=
        multiplier, num_tokens=num_tokens)
    for key, value in state_dict.items():
        diffusers_name = key.replace('proj.0', 'ff.net.0.proj')
        diffusers_name = diffusers_name.replace('proj.2', 'ff.net.2')
        updated_state_dict[diffusers_name] = value
    image_projection.load_state_dict(updated_state_dict)
    return image_projection
