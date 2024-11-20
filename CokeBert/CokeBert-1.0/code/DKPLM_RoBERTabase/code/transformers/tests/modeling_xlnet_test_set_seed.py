def set_seed(self):
    random.seed(self.seed)
    torch.manual_seed(self.seed)
