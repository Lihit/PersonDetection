import cv2
import numpy as np
import os
import sys
import copy

def SplitImage(srcpicture_path, heigth_split_num, width_split_num):
    image = cv2.imread(srcpicture_path)
    (h, w) = image.shape[0:2]
    dstpicture_path = 'ProcessedImages/SplitedImages/'
    if not os.path.exists(dstpicture_path):
        print(dstpicture_path, "does not exist, but aotomatically created!")
        os.mkdir(dstpicture_path)
    for filename in os.listdir(dstpicture_path):
        if filename is not None:
            os.remove(dstpicture_path + filename)
    for i in range(heigth_split_num):
        for j in range(width_split_num):
            filepath = os.path.join(dstpicture_path, str(
                i + 1) + str(j + 1) + '.png')
            cv2.imwrite(filepath, image[i * (h // heigth_split_num):(i + 1) * (
                h // heigth_split_num), j * (w // width_split_num):(j + 1) * (w // width_split_num)])

def BoxData_special_process(BoxData_special, rows, cols, heigth_split_num, width_split_num, piex_thresh):
    BoxData_special_return = []
    for i in range(heigth_split_num - 1):
        line_y = rows * (i + 1) // heigth_split_num
        BoxList_united = []
        for key in list(BoxData_special.keys()):
            BoxList_tmp = BoxData_special[key]
            for BoxList in BoxList_tmp:
                if BoxList == []:
                    continue
                if abs(BoxList[1] - line_y) < piex_thresh or abs(BoxList[1] + BoxList[3] - line_y) < piex_thresh:
                    BoxList_united.append(BoxList)
        [BoxList_del_return, BoxList_united_return] = BoxList_united_process_y(
            BoxList_united,piex_thresh)
        for box in BoxList_del_return:
            flag = 0
            for i in range(len(BoxList_united)):
                if BoxList_united[i] == box:
                    BoxList_united[i] = []
                    break
            for key in list(BoxData_special.keys()):
                BoxList_tmp = BoxData_special[key]
                for i in range(len(BoxList_tmp)):
                    if BoxList_tmp[i] == box:
                        BoxList_tmp[i] = []
                        break
                        flag = 1
                if flag == 1:
                    break
        BoxList_united.extend(BoxList_united_return)
        BoxData_special[key].extend(BoxList_united_return)
        BoxData_special_return.extend(BoxList_united)

    for j in range(width_split_num - 1):
        line_x = cols * (j + 1) // width_split_num
        BoxList_united = []
        for key in list(BoxData_special.keys()):
            BoxList_tmp = BoxData_special[key]
            for BoxList in BoxList_tmp:
                if BoxList == []:
                    continue
                if abs(BoxList[0] - line_x) < piex_thresh or abs(BoxList[0] + BoxList[2] - line_x) < piex_thresh:
                    BoxList_united.append(BoxList)
        [BoxList_del_return, BoxList_united_return] = BoxList_united_process_x(
            BoxList_united,piex_thresh)
        for box in BoxList_del_return:
            flag = 0
            for i in range(len(BoxList_united)):
                if BoxList_united[i] == box:
                    BoxList_united[i] = []
                    break
            for key in list(BoxData_special.keys()):
                BoxList_tmp = BoxData_special[key]
                for i in range(len(BoxList_tmp)):
                    if BoxList_tmp[i] == box:
                        BoxList_tmp[i] = []
                        break
                        flag = 1
                if flag == 1:
                    break
        BoxList_united.extend(BoxList_united_return)
        BoxData_special[key].extend(BoxList_united_return)
        BoxData_special_return.extend(BoxList_united)
    return BoxData_special


def BoxList_united_process_y(BoxList_united,piex_thresh):
    index_save = []
    BoxList_del_return = []
    BoxList_united_return = []
    BoxList_united_copy = copy.copy(BoxList_united)
    for i in range(len(BoxList_united_copy) - 1):
        BoxList_i = BoxList_united_copy[i]
        box1 = [x1, x2] = [BoxList_i[0], BoxList_i[0] + BoxList_i[2]]
        for j in range(i + 1, len(BoxList_united_copy)):
            BoxList_j = BoxList_united_copy[j]
            box2 = [x3, x4] = [BoxList_j[0], BoxList_j[0] + BoxList_j[2]]
            box3 = np.array(box1) - np.array(box2)
            if abs(box3[0]) < 10:
                box3[0] = 0
            if abs(box3[1]) < 10:
                box3[1] = 0
            if box3[0] * box3[1] <= 0:
                if i not in index_save:
                    index_save.append(i)
                if j not in index_save:
                    index_save.append(j)
                box_list = [min(BoxList_i[0], BoxList_j[0]), min(BoxList_i[1], BoxList_j[
                    1]), max(BoxList_i[2], BoxList_j[2]), BoxList_i[3] + BoxList_j[3]]
                BoxList_united_return.append(box_list)
                break
    for index in index_save:
        BoxList_del_return.append(BoxList_united[index])

    return [BoxList_del_return, BoxList_united_return]


def BoxList_united_process_x(BoxList_united,piex_thresh):
    index_save = []
    BoxList_del_return = []
    BoxList_united_return = []
    BoxList_united_copy = copy.copy(BoxList_united)
    for i in range(len(BoxList_united_copy) - 1):
        BoxList_i = BoxList_united_copy[i]
        box1 = [y1, y2] = [BoxList_i[1], BoxList_i[1] + BoxList_i[3]]
        for j in range(i + 1, len(BoxList_united_copy)):
            BoxList_j = BoxList_united_copy[j]
            box2 = [y3, y4] = [BoxList_j[1], BoxList_j[1] + BoxList_j[3]]
            box3 = np.array(box1) - np.array(box2)
            if abs(box3[0]) < 10:
                box3[0] = 0
            if abs(box3[1]) < 10:
                box3[1] = 0
            if box3[0] * box3[1] <= 0:
                if i not in index_save:
                    index_save.append(i)
                if j not in index_save:
                    index_save.append(j)
                    box_list = [min(BoxList_i[0], BoxList_j[0]), min(BoxList_i[1], BoxList_j[
                        1]), BoxList_i[2] + BoxList_j[2], max(BoxList_i[3], BoxList_j[3])]
                BoxList_united_return.append(box_list)
                break
    for index in index_save:
        BoxList_del_return.append(BoxList_united[index])

    return [BoxList_del_return, BoxList_united_return]


def UnitedImage(srcpicture_path, SplitImageDict, heigth_split_num, width_split_num, piex_thresh):
    img = cv2.imread(srcpicture_path)
    rows = int(img.shape[0])
    cols = int(img.shape[1])
    BoxDict_Perfect = {}  # 啥也不用干的
    BoxDict_Reunited = {}  # 需要做合并的
    for filename in list(SplitImageDict.keys()):
        BoxList_Perfect = []
        BoxList_Reunited = []
        name_split = filename.split('.')[0]
        i = int(name_split) // 10 - 1
        j = int(name_split) % 10 - 1
        Imagefilepath = os.path.join("ProcessedImages/SplitedImages", filename)
        BoxData = SplitImageDict[filename]
        for key in list(BoxData.keys()):
            (x, y, w, h) = BoxData[key]
            [left, right, top, bot] = [x, x + w, y, y + h]
            BoxTmp = [x + j * cols // width_split_num,
                      y + i * rows // heigth_split_num, w, h]
            if left < piex_thresh or abs(right - cols // width_split_num) < piex_thresh or top < piex_thresh or abs(bot - rows // heigth_split_num) < piex_thresh:
                BoxList_Reunited.append(BoxTmp)
                #BoxList_Perfect.append(BoxTmp)
            else:
                BoxList_Perfect.append(BoxTmp)
        BoxDict_Perfect[name_split] = BoxList_Perfect
        BoxDict_Reunited[name_split] = BoxList_Reunited

    BoxDict_Reunited_return = BoxData_special_process(
        BoxDict_Reunited, rows, cols, heigth_split_num, width_split_num, piex_thresh)
    for key in list(BoxDict_Reunited_return.keys()):
        BoxData_array_tmp = BoxDict_Reunited_return[key]
        for box_list in BoxData_array_tmp:
            if len(box_list) != 4:
                continue
            [x, y, w, h] = box_list
            if x > cols:
                continue
            if y > rows:
                continue
            if x + w > cols:
                w = cols - x
            if y + h > rows:
                h = rows - y
            BoxDict_Perfect[key].append([x, y, w, h])
    return BoxDict_Perfect
