def get_global_step(self):
    return int(self.global_step.cpu().numpy()[0])
