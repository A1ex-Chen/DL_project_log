def load(self, path, *args, **kwargs):
    need_sync = False
    if path and isinstance(self.model, DistributedDataParallel):
        logger = logging.getLogger(__name__)
        path = self.path_manager.get_local_path(path)
        has_file = os.path.isfile(path)
        all_has_file = comm.all_gather(has_file)
        if not all_has_file[0]:
            raise OSError(f'File {path} not found on main worker.')
        if not all(all_has_file):
            logger.warning(
                f'Not all workers can read checkpoint {path}. Training may fail to fully resume.'
                )
            need_sync = True
        if not has_file:
            path = None
    ret = super().load(path, *args, **kwargs)
    if need_sync:
        logger.info('Broadcasting model states from main worker ...')
        self.model._sync_params_and_buffers()
    return ret
