def test_shape_mismatch(self):
    height, width = 32, 32
    with self.assertRaises(AssertionError):
        prepare_mask_and_masked_image(torch.randn(3, height, width), torch.
            randn(64, 64), height, width, return_image=True)
    with self.assertRaises(AssertionError):
        prepare_mask_and_masked_image(torch.randn(2, 3, height, width),
            torch.randn(4, 64, 64), height, width, return_image=True)
    with self.assertRaises(AssertionError):
        prepare_mask_and_masked_image(torch.randn(2, 3, height, width),
            torch.randn(4, 1, 64, 64), height, width, return_image=True)
