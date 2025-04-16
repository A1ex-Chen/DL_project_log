def save_pretrained(self, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    if self.config.index_path is None:
        index_path = os.path.join(save_directory, 'hf_dataset_index.faiss')
        self.index.dataset.get_index(self.config.index_name).save(index_path)
        self.config.index_path = index_path
    self.config.save_pretrained(save_directory)
