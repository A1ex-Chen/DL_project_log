@property
def image_params(self) ->frozenset:
    raise NotImplementedError(
        'You need to set the attribute `image_params` in the child test class. `image_params` are tested for if all accepted input image types (i.e. `pt`,`pil`,`np`) are producing same results'
        )
