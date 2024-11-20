def _sleep(self, fSleepTime=0):
    if fSleepTime == 0:
        fSleepTime = random.uniform(1, 1.5)
    time.sleep(fSleepTime)
