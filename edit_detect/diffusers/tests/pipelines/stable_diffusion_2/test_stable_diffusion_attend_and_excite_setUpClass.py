@classmethod
def setUpClass(cls):
    super().setUpClass()
    torch.use_deterministic_algorithms(False)
