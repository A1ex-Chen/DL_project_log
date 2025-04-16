def setUp(self):
    super().setUp()
    gc.collect()
    torch.cuda.empty_cache()
