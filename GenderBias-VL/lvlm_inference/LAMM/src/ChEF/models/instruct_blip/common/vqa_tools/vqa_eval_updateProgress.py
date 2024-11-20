def updateProgress(self, progress):
    barLength = 20
    status = ''
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = 'error: progress var must be float\r\n'
    if progress < 0:
        progress = 0
        status = 'Halt...\r\n'
    if progress >= 1:
        progress = 1
        status = 'Done...\r\n'
    block = int(round(barLength * progress))
    text = '\rFinshed Percent: [{0}] {1}% {2}'.format('#' * block + '-' * (
        barLength - block), int(progress * 100), status)
    sys.stdout.write(text)
    sys.stdout.flush()
