def is_compiled_module(module):
    """Check whether the module was compiled with torch.compile()"""
    if is_torch_version('<', '2.0.0') or not hasattr(torch, '_dynamo'):
        return False
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)
