def allocate_buffers(self, shape_dict=None, device='cuda'):
    for idx in range(trt_util.get_bindings_per_profile(self.engine)):
        binding = self.engine[idx]
        if shape_dict and binding in shape_dict:
            shape = shape_dict[binding]
        else:
            shape = self.engine.get_binding_shape(binding)
        dtype = trt.nptype(self.engine.get_binding_dtype(binding))
        if self.engine.binding_is_input(binding):
            self.context.set_binding_shape(idx, shape)
        tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[
            dtype]).to(device=device)
        self.tensors[binding] = tensor
        self.buffers[binding] = cuda.DeviceView(ptr=tensor.data_ptr(),
            shape=shape, dtype=dtype)
