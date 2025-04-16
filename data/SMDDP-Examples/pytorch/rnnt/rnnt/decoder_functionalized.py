def functionalized(*inputs):
    with torch.no_grad():
        for i, arg in zip(sample_args, inputs):
            if i.data_ptr() != arg.data_ptr():
                i.copy_(arg)
    fwd_graph.replay()
    return outputs
