def _get_local_path(self, path, **kwargs):
    logger = logging.getLogger(__name__)
    catalog_path = ModelCatalog.get(path[len(self.PREFIX):])
    logger.info('Catalog entry {} points to {}'.format(path, catalog_path))
    return PathManager.get_local_path(catalog_path, **kwargs)
