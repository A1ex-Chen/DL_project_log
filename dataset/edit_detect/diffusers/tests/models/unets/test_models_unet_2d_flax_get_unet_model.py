def get_unet_model(self, fp16=False, model_id='CompVis/stable-diffusion-v1-4'):
    dtype = jnp.bfloat16 if fp16 else jnp.float32
    revision = 'bf16' if fp16 else None
    model, params = FlaxUNet2DConditionModel.from_pretrained(model_id,
        subfolder='unet', dtype=dtype, revision=revision)
    return model, params
