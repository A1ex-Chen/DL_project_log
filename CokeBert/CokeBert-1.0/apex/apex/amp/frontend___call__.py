def __call__(self, properties):
    properties.enabled = True
    properties.opt_level = 'O0'
    properties.cast_model_type = torch.float32
    properties.patch_torch_functions = False
    properties.keep_batchnorm_fp32 = None
    properties.master_weights = False
    properties.loss_scale = 1.0
    return properties
