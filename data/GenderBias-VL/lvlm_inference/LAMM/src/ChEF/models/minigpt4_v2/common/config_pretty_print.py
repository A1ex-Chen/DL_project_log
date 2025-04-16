def pretty_print(self):
    logging.info('\n=====  Running Parameters    =====')
    logging.info(self._convert_node_to_json(self.config.run))
    logging.info('\n======  Dataset Attributes  ======')
    datasets = self.config.datasets
    for dataset in datasets:
        if dataset in self.config.datasets:
            logging.info(f'\n======== {dataset} =======')
            dataset_config = self.config.datasets[dataset]
            logging.info(self._convert_node_to_json(dataset_config))
        else:
            logging.warning(f"No dataset named '{dataset}' in config. Skipping"
                )
    logging.info(f'\n======  Model Attributes  ======')
    logging.info(self._convert_node_to_json(self.config.model))
