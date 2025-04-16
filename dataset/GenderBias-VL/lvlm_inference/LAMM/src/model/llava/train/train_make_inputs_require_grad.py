def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)
