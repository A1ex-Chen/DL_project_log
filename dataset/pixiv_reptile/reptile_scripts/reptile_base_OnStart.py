def OnStart(self):
    if not os.path.exists(defines.PICTURES_PATH):
        os.makedirs(defines.PICTURES_PATH)
    self.SetCookie(defines.PIXIV_COOKIES)
    self._sendRequest(sUrl=defines.PIXIV_URL, dctHeaders=self.m_dctHeaders,
        timeout=5)
