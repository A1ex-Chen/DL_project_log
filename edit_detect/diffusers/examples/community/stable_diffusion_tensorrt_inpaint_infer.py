def infer(self, feed_dict, stream):
    start_binding, end_binding = trt_util.get_active_profile_bindings(self.
        context)
    device_buffers = copy(self.buffers)
    for name, buf in feed_dict.items():
        assert isinstance(buf, cuda.DeviceView)
        device_buffers[name] = buf
    bindings = [0] * start_binding + [buf.ptr for buf in device_buffers.
        values()]
    noerror = self.context.execute_async_v2(bindings=bindings,
        stream_handle=stream.ptr)
    if not noerror:
        raise ValueError('ERROR: inference failed.')
    return self.tensors
