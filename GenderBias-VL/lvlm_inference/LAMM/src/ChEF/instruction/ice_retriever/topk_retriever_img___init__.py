def __init__(self, train_dataset, test_dataset, seed: Optional[int]=43,
    ice_num: Optional[int]=1, model_ckpt: Optional[str]=
    'google/vit-base-patch16-224-in21k', batch_size: Optional[int]=1, **kwargs
    ) ->None:
    super().__init__(train_dataset, test_dataset, seed, ice_num)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.batch_size = batch_size
    img_list = self.get_corpus_from_dataset(test_dataset)
    gen_datalist = [get_image(image) for image in img_list]
    self.model_ckpt = model_ckpt
    self.extractor = AutoFeatureExtractor.from_pretrained(self.model_ckpt)
    self.model = AutoModel.from_pretrained(self.model_ckpt)
    self.encode_dataset = IMG_DatasetEncoder(gen_datalist, extractor=self.
        extractor)
    self.dataloader = DataLoader(self.encode_dataset, batch_size=self.
        batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in
        batch] for key in batch[0]})
    self.model = self.model.to(self.device)
    self.model.eval()
    self.index = self.create_index()
