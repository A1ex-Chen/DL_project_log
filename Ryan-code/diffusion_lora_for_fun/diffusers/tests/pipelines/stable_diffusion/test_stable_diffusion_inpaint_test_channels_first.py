def test_channels_first(self):
    height, width = 32, 32
    with self.assertRaises(AssertionError):
        prepare_mask_and_masked_image(torch.rand(height, width, 3), torch.
            rand(3, height, width), height, width, return_image=True)
