def __call__(self, aug_input) ->Transform:
    tfms = []
    for x in self.augs:
        tfm = x(aug_input)
        tfms.append(tfm)
    return TransformList(tfms)
