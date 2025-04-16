def get_caffe2_inputs(self, batched_inputs):
    """
        Convert pytorch-style structured inputs to caffe2-style inputs that
        are tuples of tensors.

        Args:
            batched_inputs (list[dict]): inputs to a detectron2 model
                in its standard format. Each dict has "image" (CHW tensor), and optionally
                "height" and "width".

        Returns:
            tuple[Tensor]:
                tuple of tensors that will be the inputs to the
                :meth:`forward` method. For existing models, the first
                is an NCHW tensor (padded and batched); the second is
                a im_info Nx3 tensor, where the rows are
                (height, width, unused legacy parameter)
        """
    return convert_batched_inputs_to_c2_format(batched_inputs, self.
        _wrapped_model.backbone.size_divisibility, self._wrapped_model.device)
