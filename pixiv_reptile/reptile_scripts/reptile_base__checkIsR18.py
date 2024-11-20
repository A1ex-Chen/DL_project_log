def _checkIsR18(self, dctInfo):
    if dctInfo['xRestrict'] == 1 or 'R-18' in dctInfo['tags']:
        return True
    return False
