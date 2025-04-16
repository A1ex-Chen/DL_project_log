def _get_local_path(self, path, **kwargs):
    name = path[len(self.PREFIX):]
    return PathManager.get_local_path(self.S3_DETECTRON2_PREFIX + name, **
        kwargs)
