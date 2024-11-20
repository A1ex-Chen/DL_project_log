def optimize(self, onnx_graph):
    opt = Optimizer(onnx_graph)
    opt.select_outputs([0])
    opt.cleanup()
    opt.fold_constants()
    opt.infer_shapes()
    opt.select_outputs([0], names=['text_embeddings'])
    opt_onnx_graph = opt.cleanup(return_onnx=True)
    return opt_onnx_graph
