def OnStart(self):
    super().OnStart()
    ctrl_common.CoutLog('开始爬取【指定关键词:{0}】'.format(WORDS))
    for iPage in range(PAGE_START, PAGE_END + 1):
        ctrl_common.CoutLog('当前进度【第{0}页】'.format(iPage))
        sSearchUrl = URL_SEARCH.format(WORDS, WORDS, iPage)
        dctPageData = self._getPageJsonData(sSearchUrl)
        self._getAnalysisDate(dctPageData)
        self.RunDownThread()
