@property
def batch_params(self) ->frozenset:
    raise NotImplementedError(
        "You need to set the attribute `batch_params` in the child test class. `batch_params` are the parameters required to be batched when passed to the pipeline's `__call__` method. `pipeline_params.py` provides some common sets of parameters such as `TEXT_TO_IMAGE_BATCH_PARAMS`, `IMAGE_VARIATION_BATCH_PARAMS`, etc... If your pipeline's set of batch arguments has minor changes from one of the common sets of batch arguments, do not make modifications to the existing common sets of batch arguments. I.e. a text to image pipeline `negative_prompt` is not batched should set the attribute as `batch_params = TEXT_TO_IMAGE_BATCH_PARAMS - {'negative_prompt'}`. See existing pipeline tests for reference."
        )
