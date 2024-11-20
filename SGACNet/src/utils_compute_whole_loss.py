def compute_whole_loss(self):
    return self.total_loss.cpu().numpy().item() / self.nr_pixels.cpu().numpy(
        ).item()
