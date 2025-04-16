def __init__(self, train_shards_path_or_url: Union[str, List[str]],
    eval_shards_path_or_url: Union[str, List[str]], num_train_examples: int,
    per_gpu_batch_size: int, global_batch_size: int, num_workers: int,
    resolution: int=256, center_crop: bool=True, random_flip: bool=False,
    shuffle_buffer_size: int=1000, pin_memory: bool=False,
    persistent_workers: bool=False, control_type: str='canny',
    feature_extractor: Optional[DPTFeatureExtractor]=None):
    if not isinstance(train_shards_path_or_url, str):
        train_shards_path_or_url = [list(braceexpand(urls)) for urls in
            train_shards_path_or_url]
        train_shards_path_or_url = list(itertools.chain.from_iterable(
            train_shards_path_or_url))
    if not isinstance(eval_shards_path_or_url, str):
        eval_shards_path_or_url = [list(braceexpand(urls)) for urls in
            eval_shards_path_or_url]
        eval_shards_path_or_url = list(itertools.chain.from_iterable(
            eval_shards_path_or_url))

    def get_orig_size(json):
        return int(json.get('original_width', 0.0)), int(json.get(
            'original_height', 0.0))
    if control_type == 'canny':
        image_transform = functools.partial(canny_image_transform,
            resolution=resolution)
    elif control_type == 'depth':
        image_transform = functools.partial(depth_image_transform,
            feature_extractor=feature_extractor, resolution=resolution)
    processing_pipeline = [wds.decode('pil', handler=wds.
        ignore_and_continue), wds.rename(image='jpg;png;jpeg;webp',
        control_image='jpg;png;jpeg;webp', text='text;txt;caption',
        orig_size='json', handler=wds.warn_and_continue), wds.map(
        filter_keys({'image', 'control_image', 'text', 'orig_size'})), wds.
        map_dict(orig_size=get_orig_size), wds.map(image_transform), wds.
        to_tuple('image', 'control_image', 'text', 'orig_size', 'crop_coords')]
    pipeline = [wds.ResampledShards(train_shards_path_or_url),
        tarfile_to_samples_nothrow, wds.select(WebdatasetFilter(min_size=
        512)), wds.shuffle(shuffle_buffer_size), *processing_pipeline, wds.
        batched(per_gpu_batch_size, partial=False, collation_fn=
        default_collate)]
    num_worker_batches = math.ceil(num_train_examples / (global_batch_size *
        num_workers))
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(
        num_worker_batches)
    self._train_dataloader = wds.WebLoader(self._train_dataset, batch_size=
        None, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers)
    self._train_dataloader.num_batches = num_batches
    self._train_dataloader.num_samples = num_samples
    pipeline = [wds.SimpleShardList(eval_shards_path_or_url), wds.
        split_by_worker, wds.tarfile_to_samples(handler=wds.
        ignore_and_continue), *processing_pipeline, wds.batched(
        per_gpu_batch_size, partial=False, collation_fn=default_collate)]
    self._eval_dataset = wds.DataPipeline(*pipeline)
    self._eval_dataloader = wds.WebLoader(self._eval_dataset, batch_size=
        None, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers)
