def CheckFileIsExists(sFilePath):
    if os.path.exists(sFilePath):
        return True
    return False
