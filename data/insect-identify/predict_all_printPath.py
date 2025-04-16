def printPath(path, level=1):
    global allFileNum
    """'' 
    打印一个目录下的所有文件夹和文件 
    level:当前目录级别
    path：当前目录路径
    """
    dirList = []
    fileList = []
    files = os.listdir(path)
    dirList.append(str(level))
    for f in files:
        if os.path.isdir(path + '/' + f):
            if f[0] == '.':
                pass
            else:
                dirList.append(f)
        if os.path.isfile(path + '/' + f):
            fileList.append(f)
    i_dl = 0
    for dl in dirList:
        if i_dl == 0:
            i_dl = i_dl + 1
        else:
            print('-' * int(dirList[0]), dl)
            printPath(path + '/' + dl, int(dirList[0]) + 1)
    for fl in fileList:
        if '.jpg' in fl or '.png' in fl or '.JPG' in fl or '.PNG' in fl:
            current_path = path + '/' + fl
            print('-' * int(dirList[0]), fl)
            predict.predict_by_file(input_image_path=current_path,
                checkpoint_file_path='checkpoint_dir\\resnet50_102improve8.pth'
                )
        allFileNum = allFileNum + 1
