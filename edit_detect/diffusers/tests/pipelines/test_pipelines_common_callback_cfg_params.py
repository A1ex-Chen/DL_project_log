@property
def callback_cfg_params(self) ->frozenset:
    raise NotImplementedError(
        "You need to set the attribute `callback_cfg_params` in the child test class that requires to run test_callback_cfg. `callback_cfg_params` are the parameters that needs to be passed to the pipeline's callback function when dynamically adjusting `guidance_scale`. They are variables that require specialtreatment when `do_classifier_free_guidance` is `True`. `pipeline_params.py` provides some common sets of parameters such as `TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS`. If your pipeline's set of cfg arguments has minor changes from one of the common sets of cfg arguments, do not make modifications to the existing common sets of cfg arguments. I.e. for inpaint pipeline, you  need to adjust batch size of `mask` and `masked_image_latents` so should set the attribute as`callback_cfg_params = TEXT_TO_IMAGE_CFG_PARAMS.union({'mask', 'masked_image_latents'})`"
        )
