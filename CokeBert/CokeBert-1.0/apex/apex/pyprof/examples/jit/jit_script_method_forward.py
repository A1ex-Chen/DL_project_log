@torch.jit.script_method
def forward(self, input):
    return self.n * input + self.m
