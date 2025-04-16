def _getAnalysisDate(self, dctPageData):
    lstPictureID = []
    if isinstance(dctPageData['body']['illusts'], dict):
        lstPictureID += list(dctPageData['body']['illusts'].keys())
    if isinstance(dctPageData['body']['manga'], dict):
        lstPictureID += list(dctPageData['body']['manga'].keys())
    self.m_dctHeaders['Referer'] = URL_PAINTER_ILLUSTS_ID.format(PAINTER_ID)
    lstInfoItems = []
    for i in tqdm(range(0, len(lstPictureID), 50)):
        ids_str = ''
        iEnd = min(len(lstPictureID), i + 50)
        for sID in lstPictureID[i:iEnd]:
            ids_str += 'ids%5B%5D={0}&'.format(sID)
        sUrl = URL_PAINTER_ILLUSTS.format(PAINTER_ID, ids_str)
        sHtml = self._sendRequest(sUrl, dctHeaders=self.m_dctHeaders,
            timeout=15).text
        dctData = json.loads(sHtml)
        dctWorks = dctData['body']['works']
        for dctInfo in dctWorks.values():
            savePath = ctrl_common.InitLocalInfo(dctInfo['userId'], dctInfo
                ['userName'])
            lstInfoItems.append({'pictureId': str(dctInfo['id']),
                'painterId': dctInfo['userId'], 'painterName': dctInfo[
                'userName'], 'title': dctInfo['title'], 'tags': dctInfo[
                'tags'], 'savePath': savePath, 'xRestrict': dctInfo[
                'xRestrict']})
    self.m_lstInfoItems = lstInfoItems
