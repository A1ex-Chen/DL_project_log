def save_pretrained(self, save_directory):
    if isinstance(self.index, CustomHFIndex):
        if self.config.index_path is None:
            index_path = os.path.join(save_directory, 'hf_dataset_index.faiss')
            self.index.dataset.get_index('embeddings').save(index_path)
            self.config.index_path = index_path
        if self.config.passages_path is None:
            passages_path = os.path.join(save_directory, 'hf_dataset')
            faiss_index = self.index.dataset._indexes.pop('embeddings')
            self.index.dataset.save_to_disk(passages_path)
            self.index.dataset._indexes['embeddings'] = faiss_index
            self.config.passages_path = passages_path
    self.config.save_pretrained(save_directory)
    rag_tokenizer = RagTokenizer(question_encoder=self.
        question_encoder_tokenizer, generator=self.generator_tokenizer)
    rag_tokenizer.save_pretrained(save_directory)
