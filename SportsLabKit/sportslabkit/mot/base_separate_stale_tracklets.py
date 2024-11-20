def separate_stale_tracklets(self, unassigned_tracklets):
    stale_tracklets, non_stale_tracklets = [], []
    for tracklet in unassigned_tracklets:
        if tracklet.is_stale():
            stale_tracklets.append(tracklet)
        else:
            non_stale_tracklets.append(tracklet)
    return non_stale_tracklets, stale_tracklets
