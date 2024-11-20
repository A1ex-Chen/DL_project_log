def export_caffe2(self):
    """
        Export the model to Caffe2's protobuf format.
        The returned object can be saved with its :meth:`.save_protobuf()` method.
        The result can be loaded and executed using Caffe2 runtime.

        Returns:
            :class:`Caffe2Model`
        """
    from .caffe2_export import export_caffe2_detection_model
    predict_net, init_net = export_caffe2_detection_model(self.
        traceable_model, self.traceable_inputs)
    return Caffe2Model(predict_net, init_net)
