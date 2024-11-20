@staticmethod
def setup_visible_devices(visible_devices: Optional[List[int]]):
    if visible_devices is None:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(it) for it in
            visible_devices])
