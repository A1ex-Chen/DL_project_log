def init_index(self):
    if not self.index_initialized:
        if self.index_path and self.index_name:
            try:
                self.dataset.add_faiss_index(column=self.index_name,
                    metric_type=self.config.metric_type, device=self.config
                    .faiss_device)
                self.index_initialized = True
            except Exception as e:
                print(e)
                logger.info('Index not initialized')
        if self.index_name in self.dataset.features:
            self.dataset.add_faiss_index(column=self.index_name)
            self.index_initialized = True
