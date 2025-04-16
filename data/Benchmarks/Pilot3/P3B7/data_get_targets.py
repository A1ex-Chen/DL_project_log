def get_targets(self, label_file):
    """Get dictionary of targets specified by user."""
    targets = np.load(os.path.join(self.root, label_file))
    tasks = {}
    if self.subsite:
        tasks['subsite'] = targets[:, 0]
    if self.laterality:
        tasks['laterality'] = targets[:, 1]
    if self.behavior:
        tasks['behavior'] = targets[:, 2]
    if self.grade:
        tasks['grade'] = targets[:, 3]
    return tasks
