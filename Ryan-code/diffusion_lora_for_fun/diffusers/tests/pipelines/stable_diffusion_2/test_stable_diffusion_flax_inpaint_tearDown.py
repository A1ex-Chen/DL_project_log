def tearDown(self):
    super().tearDown()
    gc.collect()
