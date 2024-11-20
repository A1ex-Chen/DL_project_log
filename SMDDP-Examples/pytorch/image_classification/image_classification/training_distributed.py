def distributed(self, gpu_id):
    self.model = DDP(self.model, device_ids=[gpu_id], output_device=gpu_id)
