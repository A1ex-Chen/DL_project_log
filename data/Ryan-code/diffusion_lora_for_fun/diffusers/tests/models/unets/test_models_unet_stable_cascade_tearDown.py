def tearDown(self) ->None:
    super().tearDown()
    gc.collect()
    torch.cuda.empty_cache()
