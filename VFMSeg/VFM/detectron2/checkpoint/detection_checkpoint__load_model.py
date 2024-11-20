def _load_model(self, checkpoint):
    if checkpoint.get('matching_heuristics', False):
        self._convert_ndarray_to_tensor(checkpoint['model'])
        checkpoint['model'] = align_and_update_state_dicts(self.model.
            state_dict(), checkpoint['model'], c2_conversion=checkpoint.get
            ('__author__', None) == 'Caffe2')
    incompatible = super()._load_model(checkpoint)
    model_buffers = dict(self.model.named_buffers(recurse=False))
    for k in ['pixel_mean', 'pixel_std']:
        if k in model_buffers:
            try:
                incompatible.missing_keys.remove(k)
            except ValueError:
                pass
    for k in incompatible.unexpected_keys[:]:
        if 'anchor_generator.cell_anchors' in k:
            incompatible.unexpected_keys.remove(k)
    return incompatible
