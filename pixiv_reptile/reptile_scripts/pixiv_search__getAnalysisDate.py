def _getAnalysisDate(self, dctPageData):
    lstMangaData = dctPageData['body']['illustManga']['data']
    lstInfoItems = []
    for dctInfo in tqdm(lstMangaData):
        if 'id' in dctInfo and self._checkCollection(dctInfo):
            savePath = ctrl_common.InitLocalInfo(dctInfo['userId'], dctInfo
                ['userName'])
            lstInfoItems.append({'pictureId': str(dctInfo['id']),
                'painterId': dctInfo['userId'], 'painterName': dctInfo[
                'userName'], 'title': dctInfo['title'], 'tags': dctInfo[
                'tags'], 'savePath': savePath, 'xRestrict': dctInfo[
                'xRestrict']})
    self.m_lstInfoItems = lstInfoItems
