def get_qformer(model):
    print('loading qformer')
    qformer = qformer_model_from_original_config()
    qformer_diffusers_checkpoint = (
        qformer_original_checkpoint_to_diffusers_checkpoint(model))
    load_checkpoint_to_model(qformer_diffusers_checkpoint, qformer)
    print('done loading qformer')
    return qformer
