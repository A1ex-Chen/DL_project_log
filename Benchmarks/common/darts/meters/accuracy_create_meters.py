def create_meters(self):
    """Create an average meter for each task"""
    meters = {}
    for task, _ in self.tasks.items():
        meters[task] = AverageMeter('Acc@1', ':6.2f')
    return meters
