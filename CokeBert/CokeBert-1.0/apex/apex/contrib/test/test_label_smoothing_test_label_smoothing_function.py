def test_label_smoothing_function(self):
    smoothing, padding_idx = 0.1, 0
    N, T, H = 128, 74, 32320
    iters = 10
    loss_func = label_smoothing.SoftmaxCrossEntropyLoss.apply
    for i in range(iters):
        logits, labels, half_to_float = self.gen_test_inputs(N, T, H,
            smoothing, padding_idx)
        logits.grad = None
        losses = label_smoothing_raw(logits, labels, padding_idx, smoothing)
        loss = losses.sum()
        loss.backward()
        ref_loss = loss.clone().detach()
        ref_grad = logits.grad.clone().detach()
        logits.grad = None
        losses = loss_func(logits, labels, smoothing, padding_idx,
            half_to_float)
        loss = losses.sum()
        loss.backward()
        val_loss = loss.clone().detach()
        val_grad = logits.grad.clone().detach()
        self.print_max_diff_elem(ref_grad, val_grad)
        self.assertTrue(torch.allclose(ref_loss, val_loss, atol=1e-05, rtol
            =1e-05))
        self.assertTrue(torch.allclose(ref_grad, val_grad, atol=1e-05, rtol
            =1e-05))
