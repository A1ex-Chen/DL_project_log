def patch_forward(old_fwd):

    def new_fwd(*args, **kwargs):
        output = old_fwd(*args, **kwargs)
        return applier(output, output_caster)
    return new_fwd
