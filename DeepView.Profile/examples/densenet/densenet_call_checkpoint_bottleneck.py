@torch.jit.unused
def call_checkpoint_bottleneck(self, input):

    def closure(*inputs):
        return self.bn_function(*inputs)
    return cp.checkpoint(closure, input)
