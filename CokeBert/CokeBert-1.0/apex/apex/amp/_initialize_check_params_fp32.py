def check_params_fp32(models):
    for model in models:
        for name, param in model.named_parameters():
            if param.is_floating_point():
                if 'Half' in param.type():
                    warn_or_err(
                        """Found param {} with type {}, expected torch.cuda.FloatTensor.
When using amp.initialize, you do not need to call .half() on your model
before passing it, no matter what optimization level you choose."""
                        .format(name, param.type()))
                elif not param.is_cuda:
                    warn_or_err(
                        """Found param {} with type {}, expected torch.cuda.FloatTensor.
When using amp.initialize, you need to provide a model with parameters
located on a CUDA device before passing it no matter what optimization level
you chose. Use model.to('cuda') to use the default device."""
                        .format(name, param.type()))
        if hasattr(model, 'named_buffers'):
            buf_iter = model.named_buffers()
        else:
            buf_iter = model._buffers
        for obj in buf_iter:
            if type(obj) == tuple:
                name, buf = obj
            else:
                name, buf = obj, buf_iter[obj]
            if buf.is_floating_point():
                if 'Half' in buf.type():
                    warn_or_err(
                        """Found buffer {} with type {}, expected torch.cuda.FloatTensor.
When using amp.initialize, you do not need to call .half() on your model
before passing it, no matter what optimization level you choose."""
                        .format(name, buf.type()))
                elif not buf.is_cuda:
                    warn_or_err(
                        """Found buffer {} with type {}, expected torch.cuda.FloatTensor.
When using amp.initialize, you need to provide a model with buffers
located on a CUDA device before passing it no matter what optimization level
you chose. Use model.to('cuda') to use the default device."""
                        .format(name, buf.type()))
