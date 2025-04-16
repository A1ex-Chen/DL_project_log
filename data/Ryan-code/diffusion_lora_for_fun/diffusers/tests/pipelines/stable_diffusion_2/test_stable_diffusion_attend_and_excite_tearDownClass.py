@classmethod
def tearDownClass(cls):
    super().tearDownClass()
    torch.use_deterministic_algorithms(True)
