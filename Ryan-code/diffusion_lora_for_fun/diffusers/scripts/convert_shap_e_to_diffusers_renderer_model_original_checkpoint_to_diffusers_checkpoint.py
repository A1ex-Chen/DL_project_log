def renderer_model_original_checkpoint_to_diffusers_checkpoint(model,
    checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({f'mlp.{k}': checkpoint[
        f'{RENDERER_MLP_ORIGINAL_PREFIX}.{k}'] for k in model.mlp.
        state_dict().keys()})
    diffusers_checkpoint.update({f'params_proj.{k}': checkpoint[
        f'{RENDERER_PARAMS_PROJ_ORIGINAL_PREFIX}.{k}'] for k in model.
        params_proj.state_dict().keys()})
    diffusers_checkpoint.update({'void.background': model.state_dict()[
        'void.background']})
    cases, masks = create_mc_lookup_table()
    diffusers_checkpoint.update({'mesh_decoder.cases': cases})
    diffusers_checkpoint.update({'mesh_decoder.masks': masks})
    return diffusers_checkpoint
