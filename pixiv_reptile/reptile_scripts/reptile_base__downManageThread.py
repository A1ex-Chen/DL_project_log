def _downManageThread(start, end):
    for i in range(start, end + 1):
        sDownUrl = pre[i]['url_big']
        sPictureSuffix = sDownUrl[-6:]
        sPicturePath = sPictureSuffix
        if sPictureSuffix[0] == 'p':
            sPicturePath = '_'.join([sPictureName, sPictureSuffix[1:]])
        sDownPath = os.path.join(dir, sPicturePath)
        self._downOne(sDownUrl, dctHeaders, sDownPath)
