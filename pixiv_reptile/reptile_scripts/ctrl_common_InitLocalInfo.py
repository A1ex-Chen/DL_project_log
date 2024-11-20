def InitLocalInfo(iPainterId, sPainterName):
    sSavePath = os.path.join(defines.SAVE_PATH, '_'.join([iPainterId,
        ExchangeFilePath(sPainterName)]))
    if not os.path.exists(sSavePath):
        os.makedirs(sSavePath)
    _createDB(iPainterId)
    return sSavePath
