def tearDown(self):
    super().tearDown()
    gc.collect()
    torch.cuda.empty_cache()
