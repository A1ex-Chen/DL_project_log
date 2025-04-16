def semantic_masks(self):
    for sid in self._seg_ids:
        sinfo = self._sinfo.get(sid)
        if sinfo is None or sinfo['isthing']:
            continue
        yield (self._seg == sid).numpy().astype(np.bool), sinfo
