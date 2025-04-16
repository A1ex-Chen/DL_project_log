def get_batch(self, names) ->list:
    """Get the next batch to use for calibration, as a list of device memory pointers."""
    try:
        im0s = next(self.data_iter)['img'] / 255.0
        im0s = im0s.to('cuda') if im0s.device.type == 'cpu' else im0s
        return [int(im0s.data_ptr())]
    except StopIteration:
        return None
