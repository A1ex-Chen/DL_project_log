def seed(self):
    torch.manual_seed(2809)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
