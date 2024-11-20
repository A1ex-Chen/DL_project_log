def gather(self, outputs, output_device):
    return gather(outputs, output_device, dim=self.dim)
