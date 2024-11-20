def test_swish(self):
    act = get_activation('swish')
    self.assertIsInstance(act, nn.SiLU)
    self.assertEqual(act(torch.tensor(-100, dtype=torch.float32)).item(), 0)
    self.assertNotEqual(act(torch.tensor(-1, dtype=torch.float32)).item(), 0)
    self.assertEqual(act(torch.tensor(0, dtype=torch.float32)).item(), 0)
    self.assertEqual(act(torch.tensor(20, dtype=torch.float32)).item(), 20)
