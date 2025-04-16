def graph(func_or_module, sample_args, graph_stream=None, warmup_iters=2,
    warmup_only=False):
    assert isinstance(sample_args, tuple)
    was_module = isinstance(func_or_module, torch.nn.Module)
    if was_module:
        if isinstance(func_or_module, torch.nn.parallel.DistributedDataParallel
            ):
            func_or_module = func_or_module.module
        module_params = tuple(func_or_module.parameters())
        functional_args = sample_args + module_params
    stream = torch.cuda.Stream() if graph_stream is None else graph_stream
    ambient_stream = torch.cuda.current_stream()
    stream.wait_stream(ambient_stream)
    with torch.cuda.stream(stream):
        for _ in range(warmup_iters):
            outputs = func_or_module(*sample_args)
            outputs_was_tensor = isinstance(outputs, torch.Tensor)
            outputs = (outputs,) if outputs_was_tensor else outputs
            outputs_require_grad = tuple(o for o in outputs if o.requires_grad)
            args_require_grad = tuple(i for i in functional_args if i.
                requires_grad)
            buffer_incoming_grads = tuple(torch.empty_like(o) if o.
                requires_grad else None for o in outputs)
            needed_incoming_grads = tuple(b for b in buffer_incoming_grads if
                b is not None)
            torch.cuda.nvtx.range_push('autograd.grad')
            grad_inputs = torch.autograd.grad(outputs_require_grad,
                args_require_grad, needed_incoming_grads, only_inputs=True,
                allow_unused=False)
            torch.cuda.nvtx.range_pop()
        if warmup_iters > 0:
            del (outputs, outputs_require_grad, args_require_grad,
                buffer_incoming_grads, needed_incoming_grads, grad_inputs)
        if warmup_only:
            ambient_stream.wait_stream(stream)
            return func_or_module
        fwd_graph = torch.cuda.CUDAGraph()
        fwd_graph.capture_begin()
        outputs = func_or_module(*sample_args)
        fwd_graph.capture_end()
        outputs_was_tensor = isinstance(outputs, torch.Tensor)
        outputs = (outputs,) if outputs_was_tensor else outputs
        outputs_require_grad = tuple(o for o in outputs if o.requires_grad)
        args_require_grad = tuple(i for i in functional_args if i.requires_grad
            )
        buffer_incoming_grads = tuple(torch.empty_like(o) if o.
            requires_grad else None for o in outputs)
        needed_incoming_grads = tuple(b for b in buffer_incoming_grads if b
             is not None)
        bwd_graph = torch.cuda.CUDAGraph()
        bwd_graph.capture_begin(pool=fwd_graph.pool())
        torch.cuda.nvtx.range_push('capturing autograd.grad')
        grad_inputs = torch.autograd.grad(outputs_require_grad,
            args_require_grad, needed_incoming_grads, only_inputs=True,
            allow_unused=False)
        torch.cuda.nvtx.range_pop()
        bwd_graph.capture_end()
        buffer_inputs = tuple(i.detach() for i in functional_args)
        buffer_outputs = tuple(o.detach().requires_grad_(o.requires_grad) for
            o in outputs)
        buffer_grad_inputs = []
        grad_idx = 0
        for arg in functional_args:
            if arg.requires_grad:
                buffer_grad_inputs.append(grad_inputs[grad_idx])
                grad_idx += 1
            else:
                buffer_grad_inputs.append(None)
        buffer_grad_inputs = tuple(buffer_grad_inputs)
        capture_eval = False
        if capture_eval:
            with torch.no_grad():
                func_or_module.eval()
                eval_graph = torch.cuda.CUDAGraph()
                eval_graph.capture_begin()
                eval_outputs = func_or_module(*sample_args)
                eval_graph.capture_end()
                eval_outputs_was_tensor = isinstance(eval_outputs, torch.Tensor
                    )
                eval_outputs = (eval_outputs,
                    ) if eval_outputs_was_tensor else eval_outputs
                func_or_module.train()
    ambient_stream.wait_stream(stream)


    class Graphed(torch.autograd.Function):

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
                        for i, arg in zip(buffer_inputs[0:len(sample_args)],
                            inputs[0:len(sample_args)]):
                            assert i.shape == arg.shape, "eval capture shape doesn't match run input shape"
                            if i.data_ptr() != arg.data_ptr():
                                i.copy_(arg)
                        eval_graph.replay()
                        return eval_outputs
                    else:
                        outputs = func_or_module.forward_eager(*inputs[0:
                            len(sample_args)])
                        if not isinstance(outputs, tuple):
                            outputs = outputs,
                        return outputs

        @staticmethod
        def backward(ctx, *grads):
            with torch.no_grad():
                for g, grad in zip(buffer_incoming_grads, grads):
                    if g is not None:
                        g.copy_(grad)
            bwd_graph.replay()
            return tuple(b.detach() if b is not None else b for b in
                buffer_grad_inputs)
    if was_module:

        def functionalized(self, *user_args):
            out = Graphed.apply(*(user_args + module_params))
            return out[0] if outputs_was_tensor else out
        func_or_module.forward_eager = func_or_module.forward
        func_or_module.forward = types.MethodType(functionalized,
            func_or_module)
        return func_or_module
    else:
        return Graphed.apply
