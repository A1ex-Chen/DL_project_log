def run_and_save_graph(predict_net, init_net, tensor_inputs, graph_save_path):
    """
    Run the caffe2 model on given inputs, recording the shape and draw the graph.

    predict_net/init_net: caffe2 model.
    tensor_inputs: a list of tensors that caffe2 model takes as input.
    graph_save_path: path for saving graph of exported model.
    """
    logger.info('Saving graph of ONNX exported model to {} ...'.format(
        graph_save_path))
    save_graph(predict_net, graph_save_path, op_only=False)
    logger.info('Running ONNX exported model ...')
    with ScopedWS('__ws_tmp__', True) as ws:
        ws.RunNetOnce(init_net)
        initialized_blobs = set(ws.Blobs())
        uninitialized = [inp for inp in predict_net.external_input if inp
             not in initialized_blobs]
        for name, blob in zip(uninitialized, tensor_inputs):
            ws.FeedBlob(name, blob)
        try:
            ws.RunNetOnce(predict_net)
        except RuntimeError as e:
            logger.warning('Encountered RuntimeError: \n{}'.format(str(e)))
        ws_blobs = {b: ws.FetchBlob(b) for b in ws.Blobs()}
        blob_sizes = {b: ws_blobs[b].shape for b in ws_blobs if isinstance(
            ws_blobs[b], np.ndarray)}
        logger.info('Saving graph with blob shapes to {} ...'.format(
            graph_save_path))
        save_graph(predict_net, graph_save_path, op_only=False, blob_sizes=
            blob_sizes)
        return ws_blobs
