def create_index(self):
    self.select_datalist = self.get_corpus_from_dataset(self.index_ds)
    encode_datalist = DatasetEncoder(self.select_datalist, tokenizer=self.
        tokenizer)
    co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=
        self.device)
    dataloader = DataLoader(encode_datalist, batch_size=self.batch_size,
        collate_fn=co)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(self.model.
        get_sentence_embedding_dimension()))
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
