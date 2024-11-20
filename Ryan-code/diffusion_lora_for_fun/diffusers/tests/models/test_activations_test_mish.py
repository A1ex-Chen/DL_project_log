def test_mish(self):
    act = get_activation('mish')
    self.assertIsInstance(act, nn.Mish)
    self.assertEqual(act(torch.tensor(-200, dtype=torch.float32)).item(), 0)
    self.assertNotEqual(act(torch.tensor(-1, dtype=torch.float32)).item(), 0)
    self.assertEqual(act(torch.tensor(0, dtype=torch.float32)).item(), 0)
    self.assertEqual(act(torch.tensor(20, dtype=torch.float32)).item(), 20)
