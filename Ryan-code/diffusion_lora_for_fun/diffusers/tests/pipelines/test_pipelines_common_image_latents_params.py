@property
def image_latents_params(self) ->frozenset:
    raise NotImplementedError(
        'You need to set the attribute `image_latents_params` in the child test class. `image_latents_params` are tested for if passing latents directly are producing same results'
        )
