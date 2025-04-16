def OnStart(self):
    super().OnStart()
    ctrl_common.CoutLog('开始爬取【已关注用户的作品】')
    for iPage in range(PAGE_START, PAGE_END + 1):
        ctrl_common.CoutLog('当前进度【第{0}页】'.format(iPage))
        sUrl = DYNAMIC_URL.format(iPage)
        dctPageData = self._getPageJsonData(sUrl)
        self._getAnalysisDate(dctPageData)
        self.RunDownThread()
