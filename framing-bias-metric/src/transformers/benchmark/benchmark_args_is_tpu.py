@property
def is_tpu(self):
    return is_torch_tpu_available() and self.tpu
