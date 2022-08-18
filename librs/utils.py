import copy
import os
import cv2
import numpy as np
from tqdm import tqdm
from src import util
from src.body import Body
from src.hand import Hand
from librs.configs import *

def get_x_dis(candidate, subset, idx1, idx2):
    x1, x2 = candidate[idx1, 0], candidate[idx2, 0]
    return np.abs(x1 - x2)


# 坐标变换

def cvt_pos(pos, cvt_mat_t):
    u = pos[0]
    v = pos[1]
    x = (cvt_mat_t[0][0] * u + cvt_mat_t[0][1] * v + cvt_mat_t[0][2]) / (
            cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
    y = (cvt_mat_t[1][0] * u + cvt_mat_t[1][1] * v + cvt_mat_t[1][2]) / (
            cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
    return (x, y)


def update_result(pos, mode=1):
    src = cv2.imread('./images/2-001.jpg')
    w, h = 800, 500
    srcPoint = np.float32(srcpoint[mode])  # 场景2，参考线内缘四角
    canPoint = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    Mat = cv2.getPerspectiveTransform(np.array(srcPoint), np.array(canPoint))
    # result = cv2.warpPerspective(src, Mat, (w, h))
    # cv2.imshow('res', result)
    # cv2.imwrite('./remould.png', result)
    # cv2.waitKey(0)
    x, y = cvt_pos(pos, Mat)
    true_result = x * 80 / w
    return true_result


def draw_point(img, x, y):
    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.imwrite('./img.png', img)
    return img


def get_core_person(subset, candidate, keen_y):
    all_body = []

    del_idx = []
    for obj in range(subset.shape[0]):
        idx1 = int(subset[obj, 9])  # 左膝盖
        idx2 = int(subset[obj, 12])  # 右膝盖
        idx3 = int(subset[obj, 8])  # 左腰
        idx4 = int(subset[obj, 11])  # 右腰
        idx5 = int(subset[obj, 10])  # 左脚
        idx6 = int(subset[obj, 13])  # 右脚
        if idx1 != -1:
            if candidate[idx1, 1] < keen_y:
                del_idx.append(obj)
                continue
        if idx2 != -1:
            if candidate[idx2, 1] < keen_y:
                del_idx.append(obj)
                continue
        if idx3 != -1:
            if candidate[idx3, 1] < keen_y:
                del_idx.append(obj)
                continue
        if idx4 != -1:
            if candidate[idx4, 1] < keen_y:
                del_idx.append(obj)
                continue
        if idx5 != -1:
            if candidate[idx5, 1] < keen_y:
                del_idx.append(obj)
                continue
        if idx6 != -1:
            if candidate[idx6, 1] < keen_y:
                del_idx.append(obj)
                continue
        elif idx1 == -1 and idx2 == -1 and idx3 == -1 and idx4 == -1 and idx5 == -1 and idx6 == -1:
            del_idx.append(obj)
    subset = np.delete(subset, del_idx, 0)

    # for obj in range(subset.shape[0]):
    #     idx1 = int(subset[obj, 9])
    #     idx2 = int(subset[obj, 12])
    #     print(f'save keen x,y {candidate[idx1, 0], candidate[idx1, 1]}')
    #     print(f'save keen x,y {candidate[idx2, 0], candidate[idx2, 1]}')

    if subset.shape[0] == 0:
        return []

    for obj in range(subset.shape[0]):

        left_idx1 = int(subset[obj, 8])
        left_idx2 = int(subset[obj, 10])
        if left_idx1 != -1 and left_idx2 != -1:
            left_dis = get_x_dis(candidate, subset, left_idx1, left_idx2)
        else:
            left_dis = 0

        right_idx1 = int(subset[obj, 11])
        right_idx2 = int(subset[obj, 13])
        if right_idx1 != -1 and right_idx2 != -1:
            right_dis = get_x_dis(candidate, subset, right_idx1, right_idx2)
        else:
            right_dis = 0

        dis = [left_dis, right_dis]
        body_len = np.max(dis)
        all_body.append(body_len)
    max_body = np.argmax(all_body)
    subset = subset[max_body]

    # minus = np.sum(subset == -1)
    # candidate_1 = np.ones((18 - minus, 4))
    # j = 0
    # for i in range(18):
    #     if subset[i] != -1:
    #         subset[i] = j
    #         candidate_1[j] = candidate[i]
    #         j += 1
    # candidate = candidate_1

    subset = subset.reshape(1, -1)
    # detect hand

    return subset, candidate


def cal_one_best_result(root_dir, file, pics, mode, ):
    hand_loc_list = []
    for pic in pics:
        picture = os.path.join(os.path.join(root_dir, file, pic))
        if not os.path.exists(os.path.join(save_path, str(id))):
            os.makedirs(os.path.join(save_path, str(id)))
        hand, canvas = plot_one(picture, save_file=os.path.join(save_path, str(id), f'{pic}'), keen_y=keen_y[mode])
        if len(hand) != 0:
            hand_location = hand[0][12]
        else:
            hand_location = [0, 0]
        hand_loc_list.append(hand_location)
        print(f'file {file}:{pic[:-4]}/{len(pics)} finished ')
    hand_loc_list = np.array(hand_loc_list)
    large_site = np.argmax(hand_loc_list, axis=0)[0]
    large_hand_x, large_hand_y = hand_loc_list[large_site, 0], hand_loc_list[large_site, 1]
    if ins[mode, 0, 0] < large_hand_x < ins[mode, 1, 0] and \
            bg[mode, 0, 1] < large_hand_y < bg[mode, 1, 1]:
        print('最远手的坐标是({},{})'.format(large_hand_x, large_hand_y))
        result = (large_hand_x - ins[mode, 0, 0]) / (ins[mode, 1, 0] - ins[mode, 0, 0]) * 80
        result = result.round(2)
        print('记录成绩为{}'.format(ori_data.iloc[id - 1, 4]))
        print(f'预测成绩为{result}')
        print(f'最远帧为{large_site + 1}')
    else:
        result = 0
        print('最远手的位置不在范围内')
    return large_site + 1, result
