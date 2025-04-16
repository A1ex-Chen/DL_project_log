def _unZip(self, sDownPath, sTempPath, sZipUrl, dctHeaders):
    self._downOne(sZipUrl, dctHeaders, sDownPath)
    lstTempFile = []
    with zipfile.ZipFile(sDownPath, 'r') as zip_ref:
        for sImgName in zip_ref.namelist():
            lstTempFile.append(sImgName)
        zip_ref.extractall(sTempPath)
        zip_ref.close()
    return lstTempFile
