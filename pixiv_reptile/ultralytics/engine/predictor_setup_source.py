def setup_source(self, source):
    """Sets up source and inference mode."""
    self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride,
        min_dim=2)
    self.transforms = getattr(self.model.model, 'transforms',
        classify_transforms(self.imgsz[0], crop_fraction=self.args.
        crop_fraction)) if self.args.task == 'classify' else None
    self.dataset = load_inference_source(source=source, batch=self.args.
        batch, vid_stride=self.args.vid_stride, buffer=self.args.stream_buffer)
    self.source_type = self.dataset.source_type
    if not getattr(self, 'stream', True) and (self.source_type.stream or
        self.source_type.screenshot or len(self.dataset) > 1000 or any(
        getattr(self.dataset, 'video_flag', [False]))):
        LOGGER.warning(STREAM_WARNING)
    self.vid_writer = {}
