import os
import bs4
import json
import time
import copy
import random
import shutil
import imageio
import zipfile

import torch
import urllib3
import requests
import threading
from . import defines
from . import ctrl_common
from tqdm import tqdm
from ultralytics import YOLO

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # 取消警告

class CReptileBase:       # 爬虫基类




    # region 线程下载

    # endregion











            for i in range(num_mange_thread):
                start = i * elements + min(i, remaining_elements)
                end = start + elements + (1 if i < remaining_elements else 0) - 1
                thread = threading.Thread(target=_downManageThread, args=(start, end))
                thread.start()
        else:
            # 单图
            sDownUrl = dctIllustDetials['url_big']
            sPicturePath = '_'.join([iPictureID, sPictureName, sDownUrl[-4:]])
            sDownPath = os.path.join(sSavePath, sPicturePath)
            self._downOne(sDownUrl, dctHeaders, sDownPath)
        ctrl_common.InsertPicture(iPainterId, iPictureID)

    def _downOne(self, sDownUrl, dctHeaders, sDownPath):
        # 下载单个内容
        if ctrl_common.CheckFileIsExists(sDownPath):
            return
        dctHeaders['Referer'] = sDownUrl
        oSession = self._sendRequest(sDownUrl, dctHeaders=dctHeaders, timeout=15)
        with open(sDownPath, 'ab') as file:
            file.write(oSession.content)
            file.close()
        recommend = self.yolo_check_recommend(source=sDownPath)
        if recommend:
            ctrl_common.CoutLog("{}-推荐".format(sDownPath))

    def _unZip(self, sDownPath, sTempPath, sZipUrl, dctHeaders):
        self._downOne(sZipUrl, dctHeaders, sDownPath)
        lstTempFile = []
        with zipfile.ZipFile(sDownPath, 'r') as zip_ref:
            for sImgName in zip_ref.namelist():
                lstTempFile.append(sImgName)
            zip_ref.extractall(sTempPath)
            zip_ref.close()
        return lstTempFile

    def _mergerZipGif(self, lstTempFile, dictFrams, sTempPath, sDownPath):
        sGifPath = sDownPath.replace('.zip', '.gif')
        lstImg = []
        lstDelay = []
        for sImgName in lstTempFile:
            lstDelay.append(dictFrams[sImgName])
            lstImg.append(imageio.imread(os.path.join(sTempPath, sImgName)))
        imageio.mimsave(sGifPath, lstImg, duration=lstDelay, loop=0)
        os.remove(sDownPath)
        if os.path.exists(sTempPath):
            shutil.rmtree(sTempPath)

    def _checkIsR18(self, dctInfo):
        if dctInfo['xRestrict'] == 1 or 'R-18' in dctInfo['tags']:
            return True
        return False

    def yolo_check_recommend(self, source):
        model = YOLO("../other/pose-best.pt")
        results = model.predict(
            source=source,
            imgsz=640,  # 模型训练时大小
            half=True,  # 半精度推理，加快推理速度
        )
        recommend = False
        for result in tqdm(results):
            if recommend:
                return recommend
            keypoints = result.keypoints
            if keypoints.conf.shape != torch.Size([0]):
                points = keypoints.conf[0]
                recommend = recommend or self.check_is_recommend(points)
        return recommend

    def check_is_recommend(self, points):
        head = bool(points[0] > 0.85 and (points[1] > 0.85 or points[1] > 0.85))
        return head