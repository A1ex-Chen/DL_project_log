def instance_masks(self):
    for sid in self._seg_ids:
        sinfo = self._sinfo.get(sid)
        if sinfo is None or not sinfo['isthing']:
            continue
        mask = (self._seg == sid).numpy().astype(np.bool)
        if mask.sum() > 0:
            yield mask, sinfo
