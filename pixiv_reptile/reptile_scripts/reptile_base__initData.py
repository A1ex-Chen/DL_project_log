def _initData(self):
    self.m_oSession = requests.session()
    self.m_dctHeaders = {'User-Agent': defines.REPTILE_AGENT, 'Referer':
        defines.PIXIV_URL, 'content-type':
        'application/x-www-form-urlencoded', 'Connection': 'keep-alive',
        'Cookie': ''}
    self.m_lstInfoItems = []
    self.m_lstThread = []
