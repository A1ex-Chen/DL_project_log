def test_one_hot(self):
    num_classes = 10
    inp = torch.randint(0, num_classes, (128, 16), device='cuda')
    output = F.one_hot(inp, num_classes=10)
