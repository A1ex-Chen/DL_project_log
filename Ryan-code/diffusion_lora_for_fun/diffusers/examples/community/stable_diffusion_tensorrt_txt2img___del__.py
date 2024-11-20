def __del__(self):
    [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.
        DeviceArray)]
    del self.engine
    del self.context
    del self.buffers
    del self.tensors
