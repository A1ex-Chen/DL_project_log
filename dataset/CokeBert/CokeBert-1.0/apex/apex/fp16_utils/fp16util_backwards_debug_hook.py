def backwards_debug_hook(grad):
    raise RuntimeError(
        'master_params recieved a gradient in the backward pass!')
