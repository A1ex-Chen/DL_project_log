def create_index(self):
    self.select_datalist = self.get_corpus_from_dataset(self.index_ds)
    self.select_datalist = [get_image(image) for image in self.select_datalist]
    encode_datalist = IMG_DatasetEncoder(self.select_datalist, extractor=
        self.extractor)
    dataloader = DataLoader(encode_datalist, batch_size=self.batch_size,
        collate_fn=lambda batch: {key: [dict[key] for dict in batch] for
        key in batch[0]})
    index = faiss.IndexIDMap(faiss.IndexFlatIP(self.model.config.hidden_size))
    res_list = self.forward(dataloader, process_bar=True, information=
        'Creating index for index set...')
    id_list = np.array([res['metadata']['id'] for res in res_list])
    self.embed_list = np.stack([res['embed'] for res in res_list])
    if hasattr(self.test_ds, 'dataset_name'
        ) and self.test_ds.dataset_name == 'MMBench':
        remove_list = self.test_ds.circularidx
        id_list = np.array([res['metadata']['id'] for res in res_list if 
            res['metadata']['id'] not in remove_list])
        self.embed_list = np.stack([res['embed'] for res in res_list if res
            ['metadata']['id'] not in remove_list])
    index.add_with_ids(self.embed_list, id_list)
    return index
