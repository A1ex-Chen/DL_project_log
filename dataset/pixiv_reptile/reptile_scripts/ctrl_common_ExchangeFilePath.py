def ExchangeFilePath(sFilePath):
    sFilePath = sFilePath.replace('\\', '[反斜杠]')
    sFilePath = sFilePath.replace('/', '[斜杠]')
    sFilePath = sFilePath.replace(':', '[冒号]')
    sFilePath = sFilePath.replace('*', '[星号]')
    sFilePath = sFilePath.replace('?', '[问号]')
    sFilePath = sFilePath.replace('"', '[双引号]')
    sFilePath = sFilePath.replace('<', '[小于号]')
    sFilePath = sFilePath.replace('>', '[大于号]')
    sFilePath = sFilePath.replace('|', '[竖线]')
    return sFilePath
