def _getPageJsonData(self, sUrl):
    self._sleep()
    sHtml = self.m_oSession.get(sUrl, headers=self.m_dctHeaders, timeout=5
        ).text
    return json.loads(sHtml)
