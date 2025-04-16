def __call__(self, batched_inputs):
    """
        Same input/output format as :meth:`GeneralizedRCNN.forward`
        """

    def _maybe_read_image(dataset_dict):
        ret = copy.copy(dataset_dict)
        if 'image' not in ret:
            image = read_image(ret.pop('file_name'), self.model.input_format)
            image = torch.from_numpy(np.ascontiguousarray(image.transpose(2,
                0, 1)))
            ret['image'] = image
        if 'height' not in ret and 'width' not in ret:
            ret['height'] = image.shape[1]
            ret['width'] = image.shape[2]
        return ret
    return [self._inference_one_image(_maybe_read_image(x)) for x in
        batched_inputs]
