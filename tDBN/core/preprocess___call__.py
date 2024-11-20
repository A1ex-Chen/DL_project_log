def __call__(self, db_infos):
    for prepor in self._preprocessors:
        db_infos = prepor(db_infos)
    return db_infos
