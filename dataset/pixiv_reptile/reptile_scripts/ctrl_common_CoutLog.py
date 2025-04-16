def CoutLog(sLog, bTime=True):
    sLogPath = os.path.join(defines.SAVE_PATH, '_log.ini')
    if os.path.exists(sLogPath):
        with open(sLogPath, 'r', encoding='utf-8') as file:
            lines = collections.deque(file, maxlen=defines.LOG_MAX_LENGTH)
    else:
        lines = collections.deque(maxlen=defines.LOG_MAX_LENGTH)
    sLog = ''.join([sLog, '\n'])
    if bTime == True:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sLog = ''.join([now, '      ', sLog])
    print(sLog)
    lines.append(sLog)
    with open(sLogPath, 'w', encoding='utf-8') as file:
        file.writelines(lines)
        file.close()
