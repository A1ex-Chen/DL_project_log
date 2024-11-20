def _downOne(self, sDownUrl, dctHeaders, sDownPath):
    if ctrl_common.CheckFileIsExists(sDownPath):
        return
    dctHeaders['Referer'] = sDownUrl
    oSession = self._sendRequest(sDownUrl, dctHeaders=dctHeaders, timeout=15)
    with open(sDownPath, 'ab') as file:
        file.write(oSession.content)
        file.close()
    recommend = self.yolo_check_recommend(source=sDownPath)
    if recommend:
        ctrl_common.CoutLog('{}-推荐'.format(sDownPath))
