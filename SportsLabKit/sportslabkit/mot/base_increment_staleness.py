def increment_staleness(self, tracklets):
    for i, _ in enumerate(tracklets):
        tracklets[i].staleness += 1
    return tracklets
