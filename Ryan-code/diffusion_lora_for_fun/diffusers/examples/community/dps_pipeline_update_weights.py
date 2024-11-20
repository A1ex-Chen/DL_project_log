def update_weights(self, k):
    if not torch.is_tensor(k):
        k = torch.from_numpy(k)
    for name, f in self.named_parameters():
        f.data.copy_(k)
