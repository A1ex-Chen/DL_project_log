@staticmethod
def load_protobuf(dir):
    """
        Args:
            dir (str): a directory used to save Caffe2Model with
                :meth:`save_protobuf`.
                The files "model.pb" and "model_init.pb" are needed.

        Returns:
            Caffe2Model: the caffe2 model loaded from this directory.
        """
    predict_net = caffe2_pb2.NetDef()
    with PathManager.open(os.path.join(dir, 'model.pb'), 'rb') as f:
        predict_net.ParseFromString(f.read())
    init_net = caffe2_pb2.NetDef()
    with PathManager.open(os.path.join(dir, 'model_init.pb'), 'rb') as f:
        init_net.ParseFromString(f.read())
    return Caffe2Model(predict_net, init_net)
