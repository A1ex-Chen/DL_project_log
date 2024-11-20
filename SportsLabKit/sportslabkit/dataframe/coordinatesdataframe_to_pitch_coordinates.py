def to_pitch_coordinates(self, drop=True):
    """Convert image coordinates to pitch coordinates."""
    transformed_groups = []
    for i, g in self.iter_players():
        pts = g[[(i[0], i[1], 'Lon'), (i[0], i[1], 'Lat')]].values
        x, y = cv2.perspectiveTransform(np.asarray([pts]), self.H).squeeze().T
        g[i[0], i[1], 'x'] = x
        g[i[0], i[1], 'y'] = y
        if drop:
            g.drop(columns=[(i[0], i[1], 'Lon'), (i[0], i[1], 'Lat')],
                inplace=True)
        transformed_groups.append(g)
    return self._constructor(pd.concat(transformed_groups, axis=1))
