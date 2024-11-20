def test_label_smoothing_perf(self):
    smoothing, padding_idx = 0.1, 0
    N, T, H = 128, 74, 32320
    iters = 1000
    loss_func = label_smoothing.SoftmaxCrossEntropyLoss.apply
    print()
    logits, labels, half_to_float = self.gen_test_inputs(N, T, H, smoothing,
        padding_idx)
    torch.cuda.synchronize()
    ts = time.time()
    for i in range(iters):
        logits.grad = None
        losses = label_smoothing_raw(logits, labels, padding_idx, smoothing)
        loss = losses.sum() / N
        loss.backward()
    torch.cuda.synchronize()
    print('Raw time {:.2f} s elapsed for {} iterations, norm {:.4f}'.format
        (time.time() - ts, iters, logits.grad.norm()))
    torch.cuda.synchronize()
    ts = time.time()
    for i in range(iters):
        logits.grad = None
        losses = loss_func(logits, labels, smoothing, padding_idx,
            half_to_float)
        loss = losses.sum() / N
        loss.backward()
    torch.cuda.synchronize()
    print('Opt time {:.2f} s elapsed for {} iterations, norm {:.4f}'.format
        (time.time() - ts, iters, logits.grad.norm()))
