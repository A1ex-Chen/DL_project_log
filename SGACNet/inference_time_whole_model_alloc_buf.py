def alloc_buf(engine):
    in_cpu = []
    in_gpu = []
    for i in range(engine.num_bindings - 1):
        shape = trt.volume(engine.get_binding_shape(i))
        dtype = trt.nptype(engine.get_binding_dtype(i))
        in_cpu.append(cuda.pagelocked_empty(shape, dtype))
        in_gpu.append(cuda.mem_alloc(in_cpu[-1].nbytes))
    shape = trt.volume(engine.get_binding_shape(engine.num_bindings - 1))
    dtype = trt.nptype(engine.get_binding_dtype(engine.num_bindings - 1))
    out_cpu = cuda.pagelocked_empty(shape, dtype)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)
    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream
