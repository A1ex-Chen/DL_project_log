def SetCookie(self, sCookie):
    self.m_dctHeaders['Cookie'] = ''.join([defines.COOKIE_HEAD, sCookie])
