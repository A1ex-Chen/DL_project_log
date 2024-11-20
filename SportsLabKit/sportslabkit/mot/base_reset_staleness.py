def reset_staleness(self, tracklets):
    for i, _ in enumerate(tracklets):
        tracklets[i].staleness = 0
    return tracklets
