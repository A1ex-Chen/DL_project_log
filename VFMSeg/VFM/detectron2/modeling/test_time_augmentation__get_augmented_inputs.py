def _get_augmented_inputs(self, input):
    augmented_inputs = self.tta_mapper(input)
    tfms = [x.pop('transforms') for x in augmented_inputs]
    return augmented_inputs, tfms
