def __init__(self, vis_processor, text_processor, location):
    super().__init__(vis_processor=vis_processor, text_processor=text_processor
        )
    self.inner_dataset = wds.DataPipeline(wds.ResampledShards(location),
        wds.tarfile_to_samples(handler=wds.warn_and_continue), wds.shuffle(
        1000, handler=wds.warn_and_continue), wds.decode('pilrgb', handler=
        wds.warn_and_continue), wds.to_tuple('jpg', 'json', handler=wds.
        warn_and_continue), wds.map_tuple(self.vis_processor, handler=wds.
        warn_and_continue), wds.map(self.to_dict, handler=wds.
        warn_and_continue))
