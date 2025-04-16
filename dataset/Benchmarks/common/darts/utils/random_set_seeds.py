def set_seeds(self):
    os.environ['PYTHONHASHSEED'] = str(self.s.pythonhash)
    random.seed(self.s.pythonrand)
    np.random.seed(self.s.numpy)
    torch.random.manual_seed(self.s.torch)
