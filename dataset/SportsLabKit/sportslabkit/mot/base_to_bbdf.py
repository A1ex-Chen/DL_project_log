def to_bbdf(self):
    """Create a bounding box dataframe."""
    all_tracklets = self.alive_tracklets + self.dead_tracklets
    return pd.concat([t.to_bbdf() for t in all_tracklets], axis=1).sort_index()
