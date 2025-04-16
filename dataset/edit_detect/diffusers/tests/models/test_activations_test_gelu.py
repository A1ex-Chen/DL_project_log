def test_gelu(self):
    act = get_activation('gelu')
    self.assertIsInstance(act, nn.GELU)
    self.assertEqual(act(torch.tensor(-100, dtype=torch.float32)).item(), 0)
    self.assertNotEqual(act(torch.tensor(-1, dtype=torch.float32)).item(), 0)
    self.assertEqual(act(torch.tensor(0, dtype=torch.float32)).item(), 0)
    self.assertEqual(act(torch.tensor(20, dtype=torch.float32)).item(), 20)
