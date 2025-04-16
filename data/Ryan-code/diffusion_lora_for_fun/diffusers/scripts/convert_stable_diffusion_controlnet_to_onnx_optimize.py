def optimize(onnx_graph, name, verbose):
    opt = Optimizer(onnx_graph, verbose=verbose)
    opt.info(name + ': original')
    opt.cleanup()
    opt.info(name + ': cleanup')
    opt.fold_constants()
    opt.info(name + ': fold constants')
    onnx_opt_graph = opt.cleanup(return_onnx=True)
    opt.info(name + ': finished')
    return onnx_opt_graph
