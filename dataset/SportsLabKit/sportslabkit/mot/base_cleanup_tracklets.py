def cleanup_tracklets(self, tracklets):
    for i, _ in enumerate(tracklets):
        tracklets[i].cleanup()

    def filter_short_tracklets(tracklet):
        return len(tracklet) >= self.min_length
    tracklets = list(filter(filter_short_tracklets, tracklets))
    return tracklets
