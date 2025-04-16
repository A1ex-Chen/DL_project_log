def _getPicture(self, dctInfo):
    sSavePath = dctInfo['savePath']
    iPictureID = dctInfo['pictureId']
    iPainterId = dctInfo['painterId']
    if ctrl_common.CheckHavePicture(iPainterId, iPictureID):
        return
    dctHeaders = copy.deepcopy(self.m_dctHeaders)
    dctHeaders['Referer'] = defines.PICTURE_URL.format(iPictureID)
    self._sleep()
    sPictureAjaxUrl = defines.PICTURE_AJAX_URL.format(iPictureID)
    oAjaxData = self._sendRequest(sPictureAjaxUrl, dctHeaders=dctHeaders,
        timeout=15)
    oSoup = bs4.BeautifulSoup(oAjaxData.text, 'html.parser')
    dctBody = json.loads(str(oSoup))['body']
    if not dctBody:
        self._sleep()
        self._getPicture(dctInfo)
        return
    dctIllustDetials = json.loads(str(oSoup))['body']['illust_details']
    bGif = bool(dctIllustDetials.get('ugoira_meta'))
    bManga = bool(dctIllustDetials.get('manga_a'))
    sPictureName = ctrl_common.ExchangeFilePath(dctInfo['title'])
    dctHeaders = copy.deepcopy(self.m_dctHeaders)
    if bGif:
        dir = os.path.join(sSavePath, 'gif')
        if not os.path.exists(dir):
            os.makedirs(dir)
        sZipUrl = dctIllustDetials['ugoira_meta']['src']
        sZipPath = '_'.join([iPictureID, sPictureName, sZipUrl[-4:]])
        dictFrams = {d['file']: d['delay'] for d in dctIllustDetials[
            'ugoira_meta']['frames']}
        sDownPath = os.path.join(dir, sZipPath)
        sTempPath = os.path.join(dir, 'temp_{}'.format(iPictureID))
        lstTempFile = self._unZip(sDownPath, sTempPath, sZipUrl, dctHeaders)
        self._mergerZipGif(lstTempFile, dictFrams, sTempPath, sDownPath)
    elif bManga:
        pre = dctIllustDetials['manga_a']
        dir = os.path.join(sSavePath, '_'.join([iPictureID, sPictureName]))
        if not os.path.exists(dir):
            os.makedirs(dir)
        num_len = len(pre)
        num_mange_thread = (defines.MANGE_NUM_THREAD if num_len > defines.
            MANGE_NUM_THREAD else num_len)
        elements = num_len // num_mange_thread
        remaining_elements = num_len % num_mange_thread

        def _downManageThread(start, end):
            for i in range(start, end + 1):
                sDownUrl = pre[i]['url_big']
                sPictureSuffix = sDownUrl[-6:]
                sPicturePath = sPictureSuffix
                if sPictureSuffix[0] == 'p':
                    sPicturePath = '_'.join([sPictureName, sPictureSuffix[1:]])
                sDownPath = os.path.join(dir, sPicturePath)
                self._downOne(sDownUrl, dctHeaders, sDownPath)
        for i in range(num_mange_thread):
            start = i * elements + min(i, remaining_elements)
            end = start + elements + (1 if i < remaining_elements else 0) - 1
            thread = threading.Thread(target=_downManageThread, args=(start,
                end))
            thread.start()
    else:
        sDownUrl = dctIllustDetials['url_big']
        sPicturePath = '_'.join([iPictureID, sPictureName, sDownUrl[-4:]])
        sDownPath = os.path.join(sSavePath, sPicturePath)
        self._downOne(sDownUrl, dctHeaders, sDownPath)
    ctrl_common.InsertPicture(iPainterId, iPictureID)
