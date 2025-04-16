def _getAnalysisDate(self, dctPageData):
    dctIllust = dctPageData['body']['thumbnails']['illust']
    lstInfoItems = []
    for dctInfo in tqdm(dctIllust):
        savePath = ctrl_common.InitLocalInfo(dctInfo['userId'], dctInfo[
            'userName'])
        lstInfoItems.append({'pictureId': str(dctInfo['id']), 'painterId':
            dctInfo['userId'], 'painterName': dctInfo['userName'], 'title':
            dctInfo['title'], 'tags': dctInfo['tags'], 'savePath': savePath,
            'xRestrict': dctInfo['xRestrict']})
    self.m_lstInfoItems = lstInfoItems
