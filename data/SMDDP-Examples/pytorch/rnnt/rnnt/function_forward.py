@staticmethod
def forward(ctx, *inputs):
    if func_or_module.training:
        with torch.no_grad():
            for i, arg in zip(buffer_inputs, inputs):
                if i.data_ptr() != arg.data_ptr():
                    i.copy_(arg)
        fwd_graph.replay()
        return buffer_outputs
    else:
        with torch.no_grad():
            if capture_eval:
                for i, arg in zip(buffer_inputs[0:len(sample_args)], inputs
                    [0:len(sample_args)]):
                    assert i.shape == arg.shape, "eval capture shape doesn't match run input shape"
                    if i.data_ptr() != arg.data_ptr():
                        i.copy_(arg)
                eval_graph.replay()
                return eval_outputs
            else:
                outputs = func_or_module.forward_eager(*inputs[0:len(
                    sample_args)])
                if not isinstance(outputs, tuple):
                    outputs = outputs,
                return outputs
