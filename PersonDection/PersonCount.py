from PersonDetect import *
from ImageProcess import *
import os
import cv2
import numpy as np
import shutil


def PersonCount(ImageName, ProbThre=0.5, nms=0.5, is_split=False, subImageLen = 200):
    originImageName = ImageName
    srcpicture_path = "OriginalImages/" + originImageName
    dstpictureDir = 'ProcessedImages/'
    for fileDir in os.listdir(dstpictureDir):
        shutil.rmtree(os.path.join(dstpictureDir, fileDir))
        os.mkdir(os.path.join(dstpictureDir, fileDir))
    image_ori = cv2.imread(srcpicture_path)
    image_ori_copy = image_ori.copy()
    piex_thresh = 10
    if is_split:
        subImageLen = 200
        heigth_split_num = image_ori.shape[0] // subImageLen
        width_split_num = image_ori.shape[1] // subImageLen
    else:
        heigth_split_num = 1
        width_split_num = 1
    # 分割图片
    p_nms = nms
    SplitImage(srcpicture_path, heigth_split_num, width_split_num)
    SplitImageDict = {}
    for i in range(heigth_split_num):
        for j in range(width_split_num):
            imgsplitPath = os.path.join(
                dstpictureDir, "SplitedImages", str(i + 1) + str(j + 1) + ".png")
            personDict = detectPerson(
                imgsplitPath, thresh=ProbThre, hier_thresh=.5, nms=p_nms)
            SplitImageDict[str(i + 1) + str(j + 1) + ".png"] = personDict
    # 将分割的图片合并起来
    BoxDict_Perfect = UnitedImage(
        srcpicture_path, SplitImageDict, heigth_split_num, width_split_num, piex_thresh)
    personNum = 0
    image_ori_copy = image_ori.copy()
    for key in list(BoxDict_Perfect.keys()):
        BoxList_Perfect = BoxDict_Perfect[key]
        for box in BoxList_Perfect:
            [x, y, w, h] = box
            cv2.rectangle(image_ori_copy, (int(x), int(y)),
                          (int(x + w), int(y + h)), (0, 0, 255), 1)
            personNum += 1
    cv2.imwrite(os.path.join("Result", originImageName.split(
        '.')[0] + '.png'), image_ori_copy)
    return personNum
