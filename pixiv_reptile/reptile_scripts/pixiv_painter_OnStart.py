def OnStart(self):
    super().OnStart()
    ctrl_common.CoutLog('开始爬取【指定画师:{0}】'.format(PAINTER_ID))
    sPainterUrl = URL_PAINTER_ILLUSTS_ID.format(PAINTER_ID)
    dctPageData = self._getPageJsonData(sPainterUrl)
    self._getAnalysisDate(dctPageData)
    self.RunDownThread()
