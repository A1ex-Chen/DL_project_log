def _create_target(self, arry):
    """Convert target dictionary"""
    target = {'site': arry[:, 0], 'subsite': arry[:, 1], 'laterality': arry
        [:, 2], 'histology': arry[:, 3], 'behaviour': arry[:, 4], 'grade':
        arry[:, 5]}
    return {task: torch.tensor(arry, dtype=torch.long) for task, arry in
        target.items()}
