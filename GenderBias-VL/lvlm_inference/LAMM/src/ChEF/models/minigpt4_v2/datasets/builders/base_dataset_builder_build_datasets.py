def build_datasets(self):
    if is_main_process():
        self._download_data()
    if is_dist_avail_and_initialized():
        dist.barrier()
    logging.info('Building datasets...')
    datasets = self.build()
    return datasets
