def tearDown(self):
    super().tearDown()
    gc.collect()
    backend_empty_cache(torch_device)
