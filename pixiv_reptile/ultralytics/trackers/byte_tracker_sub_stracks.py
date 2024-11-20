@staticmethod
def sub_stracks(tlista, tlistb):
    """DEPRECATED CODE in https://github.com/ultralytics/ultralytics/pull/1890/
        stracks = {t.track_id: t for t in tlista}
        for t in tlistb:
            tid = t.track_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())
        """
    track_ids_b = {t.track_id for t in tlistb}
    return [t for t in tlista if t.track_id not in track_ids_b]
