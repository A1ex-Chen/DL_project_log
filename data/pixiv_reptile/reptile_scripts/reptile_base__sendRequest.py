def _sendRequest(self, sUrl, dctHeaders=None, timeout=15):
    if dctHeaders is None:
        dctHeaders = self.m_dctHeaders
    iRestartTime = defines.RESTART_TIMES
    while True:
        try:
            oResponse = self.m_oSession.get(sUrl, headers=dctHeaders,
                timeout=timeout)
            return oResponse
        except Exception as e:
            print('iRestartTime', iRestartTime)
            iRestartTime -= 1
            if iRestartTime < 0:
                raise Exception('err-url:{0}'.format(sUrl))
            self._sleep(0.5)
