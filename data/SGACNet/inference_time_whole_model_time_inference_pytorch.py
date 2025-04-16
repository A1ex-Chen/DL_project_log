def time_inference_pytorch(model, inputs, device, n_runs_warmup=5):
    timings = []
    with torch.no_grad():
        outs = []
        for i in range(len(inputs[0])):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            inputs_gpu = [inp[i].to(device) for inp in inputs]
            out_pytorch = model(*inputs_gpu)
            out_pytorch = out_pytorch.cpu()
            end.record()
            torch.cuda.synchronize()
            if i >= n_runs_warmup:
                timings.append(start.elapsed_time(end) / 1000.0)
            outs.append(out_pytorch)
    return np.array(timings), outs
