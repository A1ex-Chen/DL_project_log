@property
def params(self) ->frozenset:
    raise NotImplementedError(
        "You need to set the attribute `params` in the child test class. `params` are checked for if all values are present in `__call__`'s signature. You can set `params` using one of the common set of parameters defined in `pipeline_params.py` e.g., `TEXT_TO_IMAGE_PARAMS` defines the common parameters used in text to  image pipelines, including prompts and prompt embedding overrides.If your pipeline's set of arguments has minor changes from one of the common sets of arguments, do not make modifications to the existing common sets of arguments. I.e. a text to image pipeline with non-configurable height and width arguments should set the attribute as `params = TEXT_TO_IMAGE_PARAMS - {'height', 'width'}`. See existing pipeline tests for reference."
        )
