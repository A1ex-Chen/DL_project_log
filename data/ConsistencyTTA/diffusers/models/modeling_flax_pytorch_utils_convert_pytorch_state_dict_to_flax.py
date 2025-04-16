def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model, init_key=42):
    pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}
    random_flax_params = flax_model.init_weights(PRNGKey(init_key))
    random_flax_state_dict = flatten_dict(random_flax_params)
    flax_state_dict = {}
    for pt_key, pt_tensor in pt_state_dict.items():
        renamed_pt_key = rename_key(pt_key)
        pt_tuple_key = tuple(renamed_pt_key.split('.'))
        flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key,
            pt_tensor, random_flax_state_dict)
        if flax_key in random_flax_state_dict:
            if flax_tensor.shape != random_flax_state_dict[flax_key].shape:
                raise ValueError(
                    f'PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape {random_flax_state_dict[flax_key].shape}, but is {flax_tensor.shape}.'
                    )
        flax_state_dict[flax_key] = jnp.asarray(flax_tensor)
    return unflatten_dict(flax_state_dict)
