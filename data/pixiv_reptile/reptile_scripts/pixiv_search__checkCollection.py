def _checkCollection(self, dctInfo):
    iCollection = COLLECTION_THRESHOLD
    if self._checkIsR18(dctInfo):
        iCollection += R18_COLLECTION_THRESHOLD
    sDetailsUrl = URL_DETAIL.format(dctInfo['id'])
    dctHeaders = copy.deepcopy(self.m_dctHeaders)
    dctHeaders['Referer'] = sDetailsUrl
    sHtml = self._sendRequest(sDetailsUrl, dctHeaders=dctHeaders, timeout=15
        ).text
    soup = BeautifulSoup(sHtml, 'html.parser')
    try:
        sPictureCollection = soup.find_all('span', {'class': 'count-badge'})[0
            ].text
        if sPictureCollection[-1] == '人':
            return int(sPictureCollection[0:-1]) >= iCollection
    except:
        pass
    return False
