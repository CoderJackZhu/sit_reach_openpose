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
        return [], []

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


def plot_one(test_image, save_file, keen_y=445, origin=False):
    body_estimation = Body('./librs/model/body_pose_model.pth')
    hand_estimation = Hand('./librs/model/hand_pose_model.pth')
    oriImg = cv2.imread(test_image)  # B,G,R order

    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    # original draw body pose

    if not origin:
        subset, candidate = get_core_person(subset, candidate, keen_y)
        if len(subset) == 0:
            return [], []
    hands_list = util.handDetect(candidate, subset, oriImg)
    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # if is_left:
        # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
        # plt.show()
        peaks = hand_estimation(oriImg[y:y + w, x:x + w, :])
        peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
        peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
        # else:
        #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
        #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
        #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        #     print(peaks)
        all_hand_peaks.append(peaks)
    goal = []
    for i in range(len(all_hand_peaks)):
        if all_hand_peaks[i][12].any() != 0:
            goal.append(all_hand_peaks[i])
    goal = np.array(goal)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    canvas = util.draw_handpose(canvas, goal)
    # canvas = util.draw_handpose(canvas, all_hand_peaks)
    if save_file is not None:
        cv2.imwrite(save_file, canvas)

    # cv2.imshow('result', canvas)
    # cv2.waitKey(0)
    return goal, canvas


def cal_one_best_result(data, root_dir, file, pics, mode, save_path):
    all_pic_list = []
    pic_nums = [int(pic[:-4]) for pic in pics]
    num_count = 0
    while (1):
        pic_list = []
        for pic_num in pic_nums:
            multi_list = [pic_num - num_count, pic_num, pic_num + num_count]
            for one_pic in multi_list:
                if one_pic not in all_pic_list:
                    all_pic_list.append(one_pic)
                    pic_list.append(one_pic)
        print(pic_list)
        pics = [str(pic).zfill(3) + '.jpg' for pic in pic_list]
        hand_loc_list, canvas_list = [], []
        for pic in pics:
            picture = os.path.join(os.path.join(root_dir, file, pic))
            if not os.path.exists(os.path.join(save_path, file)):
                os.mkdir(os.path.join(save_path, file))
            hand, canvas = plot_one(picture, save_file=os.path.join(save_path, file, f'{pic}'), keen_y=keen_y[mode])
            if len(hand) != 0:
                hand_location = hand[0][12]
            else:
                hand_location = [0, 0]
            hand_loc_list.append(hand_location)
            canvas_list.append(canvas)
            print(f'{file}:{pic[:-4]}/{len(pics)} finished ')
        hand_loc_list = np.array(hand_loc_list)
        if np.any(hand_loc_list != 0):
            break
        else:
            num_count += 1
    # if np.any(hand_loc_list != 0):
    large_site = np.argmax(hand_loc_list, axis=0)[0]
    large_frame = int(pics[large_site][:-4])
    best_canvas = canvas_list[large_site]
    large_hand_x, large_hand_y = hand_loc_list[large_site, 0], hand_loc_list[large_site, 1]
    if bg[mode, 0, 0] < large_hand_x < bg[mode, 1, 0] and \
            bg[mode, 0, 1] < large_hand_y < bg[mode, 1, 1]:
        print('最远手的坐标是({},{})'.format(large_hand_x, large_hand_y))
        result = (large_hand_x - ins[mode, 0, 0]) / (ins[mode, 1, 0] - ins[mode, 0, 0]) * 80
        result = result.round(2)
        print('记录成绩为{}'.format(data[4]))
        print(f'预测成绩为{result}')
        # if len(pics) > 5:
        print(f'最远帧为{large_frame}')
        return best_canvas, large_frame, [large_hand_x, large_hand_y], result
    else:
        print('最远手的坐标不在范围内')
        return 0, 0, [0, 0], 0
    # else:
    #     print('未检测到手')
    #     return -1, -1, [-1, -1], -1


def get_TAL_frame(idx):
    data = np.genfromtxt('./show_result/sit_result_TAL.txt', delimiter=' ')
    return data[idx, 1]


def plot_multi(output_dir='./test2', result_dir='./result2'):
    files = os.listdir(output_dir)
    for file in tqdm(files):
        if not os.path.exists(os.path.join(result_dir, file)):
            os.mkdir(os.path.join(result_dir, file))
        pics = os.listdir(os.path.join(output_dir, file))
        for pic in tqdm(pics):
            picture = os.path.join(os.path.join(output_dir, file, pic))
            print(picture)
            # plot_one(test_image=picture, save_file=os.path.join(result_dir, file, pic), keen_y=470)
            plot_one(test_image=picture, save_file=os.path.join(result_dir, file, pic))
