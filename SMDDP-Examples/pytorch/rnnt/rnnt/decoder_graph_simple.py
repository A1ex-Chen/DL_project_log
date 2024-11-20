def graph_simple(func_or_module, sample_args, graph_stream=None,
    warmup_iters=2, warmup_only=False):
    assert isinstance(sample_args, tuple)
    stream = torch.cuda.Stream() if graph_stream is None else graph_stream
    ambient_stream = torch.cuda.current_stream()
    stream.wait_stream(ambient_stream)
    with torch.cuda.stream(stream):
        for _ in range(warmup_iters):
            outputs = func_or_module(*sample_args)
        if warmup_iters > 0:
            del outputs
        fwd_graph = torch.cuda.CUDAGraph()
        fwd_graph.capture_begin()
        outputs = func_or_module(*sample_args)
        fwd_graph.capture_end()
    ambient_stream.wait_stream(stream)

    def functionalized(*inputs):
        with torch.no_grad():
            for i, arg in zip(sample_args, inputs):
                if i.data_ptr() != arg.data_ptr():
                    i.copy_(arg)
        fwd_graph.replay()
        return outputs
    return functionalized
