def save_protobuf(self, output_dir):
    """
        Save the model as caffe2's protobuf format.
        It saves the following files:

            * "model.pb": definition of the graph. Can be visualized with
              tools like `netron <https://github.com/lutzroeder/netron>`_.
            * "model_init.pb": model parameters
            * "model.pbtxt": human-readable definition of the graph. Not
              needed for deployment.

        Args:
            output_dir (str): the output directory to save protobuf files.
        """
    logger = logging.getLogger(__name__)
    logger.info('Saving model to {} ...'.format(output_dir))
    if not PathManager.exists(output_dir):
        PathManager.mkdirs(output_dir)
    with PathManager.open(os.path.join(output_dir, 'model.pb'), 'wb') as f:
        f.write(self._predict_net.SerializeToString())
    with PathManager.open(os.path.join(output_dir, 'model.pbtxt'), 'w') as f:
        f.write(str(self._predict_net))
    with PathManager.open(os.path.join(output_dir, 'model_init.pb'), 'wb'
        ) as f:
        f.write(self._init_net.SerializeToString())
