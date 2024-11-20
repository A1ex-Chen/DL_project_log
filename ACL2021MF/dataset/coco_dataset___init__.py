def __init__(self, config, h5_path, tokenizer, copy_vocab, attachable_index,
    caption_path=None, copy_h5_path=None, is_training=False, in_memory=
    False, cbs_class_path=None):
    if caption_path is not None:
        self._captions_reader = CocoCaptionsReader(caption_path, config.
            word_norm_jsonpath if len(config.word_norm_jsonpath) > 0 else
            None, rm_dumplicated_caption=config.rm_dumplicated_caption,
            shuffle=config.shuffle_data, is_train=is_training,
            rm_punctuation=config.rm_punctuation)
    else:
        self._captions_reader = None
    np.set_printoptions(threshold=sys.maxsize)
    self._image_features_reader = ImageFeaturesReader(h5_path)
    if config.use_copy_obj:
        self._copy_image_features_reader = ImageFeaturesReader(copy_h5_path,
            start_index=1601)
    self.config = config
    self.is_training = is_training
    self.copy_vocab = copy_vocab
    self.tokenizer = tokenizer
    self.attachable_index = attachable_index
    self.cbs_class = None
    if cbs_class_path is not None:
        self.cbs_class = {}
        with open(cbs_class_path) as out:
            for line in out:
                line = line.strip()
                items = line.split(',')
                self.cbs_class[int(items[0])] = sorted([int(v) for v in
                    items[1:]])
    self._image_ids = sorted(list(self._image_features_reader._map.keys()))
    self.obj_cache = {}
    self.cap_cache = {}
    self.global_obj_cache = {}
    if len(config.object_blacklist_path) > 0:
        with open(config.object_blacklist_path) as out:
            blacklist = json.load(out)
            full_list = blacklist['blacklist_categories'] + (blacklist[
                'val_blacklist_categories'] if not is_training else [])
        self._blacklist_categories = set([s.lower() for s in full_list])
    else:
        self._blacklist_categories = None
    self.img_index = self.tokenizer('<unk>', return_tensors='np')['input_ids'][
        0, 0]
    self.background_index = self.tokenizer('background', return_tensors='np')[
        'input_ids'][0, 0]
    if in_memory or not is_training:
        self._image_features_reader.open_h5_file()
        if config.use_copy_obj:
            self._copy_image_features_reader.open_h5_file()
        for index in tqdm(range(len(self._captions_reader))):
            img_id, cap, _ = self._captions_reader[index]
            if img_id not in self.obj_cache:
                self.process_obj(img_id)
            if self._captions_reader is not None:
                if img_id not in self.cap_cache or cap not in self.cap_cache[
                    img_id]:
                    self.process_cap(img_id, cap)
        for img_id in tqdm(self.cap_cache):
            self.process_global_cap(img_id)
        self._image_features_reader.close_h5_file()
        if config.use_copy_obj:
            self._copy_image_features_reader.close_h5_file()
