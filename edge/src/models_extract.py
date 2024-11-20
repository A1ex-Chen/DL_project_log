def extract(self, im):
    im = self.preprocess(im)
    cuda.memcpy_htod_async(self.d_input, im, self.stream)
    self.context.execute_async_v2(self.bindings, self.stream.handle, None)
    cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
    self.stream.synchronize()
    pred = self.output
    return pred.astype(np.float32)[0]
