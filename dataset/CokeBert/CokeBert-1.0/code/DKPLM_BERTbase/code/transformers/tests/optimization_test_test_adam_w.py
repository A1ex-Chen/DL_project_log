def test_adam_w(self):
    w = torch.tensor([0.1, -0.2, -0.1], requires_grad=True)
    target = torch.tensor([0.4, 0.2, -0.5])
    criterion = torch.nn.MSELoss()
    optimizer = AdamW(params=[w], lr=0.2, weight_decay=0.0)
    for _ in range(100):
        loss = criterion(w, target)
        loss.backward()
        optimizer.step()
        w.grad.detach_()
        w.grad.zero_()
    self.assertListAlmostEqual(w.tolist(), [0.4, 0.2, -0.5], tol=0.01)
