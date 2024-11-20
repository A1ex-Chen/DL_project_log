def _mergerZipGif(self, lstTempFile, dictFrams, sTempPath, sDownPath):
    sGifPath = sDownPath.replace('.zip', '.gif')
    lstImg = []
    lstDelay = []
    for sImgName in lstTempFile:
        lstDelay.append(dictFrams[sImgName])
        lstImg.append(imageio.imread(os.path.join(sTempPath, sImgName)))
    imageio.mimsave(sGifPath, lstImg, duration=lstDelay, loop=0)
    os.remove(sDownPath)
    if os.path.exists(sTempPath):
        shutil.rmtree(sTempPath)
