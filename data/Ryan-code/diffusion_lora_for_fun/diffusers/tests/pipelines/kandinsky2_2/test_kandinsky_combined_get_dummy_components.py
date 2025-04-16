def get_dummy_components(self):
    dummy = InpaintDummies()
    prior_dummy = PriorDummies()
    components = dummy.get_dummy_components()
    components.update({f'prior_{k}': v for k, v in prior_dummy.
        get_dummy_components().items()})
    return components
