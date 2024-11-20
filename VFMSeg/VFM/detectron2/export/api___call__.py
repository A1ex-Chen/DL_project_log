def __call__(self, inputs):
    """
        An interface that wraps around a Caffe2 model and mimics detectron2's models'
        input/output format. See details about the format at :doc:`/tutorials/models`.
        This is used to compare the outputs of caffe2 model with its original torch model.

        Due to the extra conversion between Pytorch/Caffe2, this method is not meant for
        benchmark. Because of the conversion, this method also has dependency
        on detectron2 in order to convert to detectron2's output format.
        """
    if self._predictor is None:
        self._predictor = ProtobufDetectionModel(self._predict_net, self.
            _init_net)
    return self._predictor(inputs)
