import copy
import os
import cv2
import numpy as np
from tqdm import tqdm
from src import util
from src.body import Body
from src.hand import Hand
from librs.utils import get_core_person


# def plot_origin(test_image='E:/Project/Sit_and_reach_clip/20220628014/109.jpg', save_file='./test.png'):
#     body_estimation = Body('./librs/model/body_pose_model.pth')
#     hand_estimation = Hand('./librs/model/hand_pose_model.pth')
#
#     oriImg = cv2.imread(test_image)  # B,G,R order
#     candidate, subset = body_estimation(oriImg)
#     canvas = copy.deepcopy(oriImg)
#     canvas = util.draw_bodypose(canvas, candidate, subset)
#     # detect hand
#     hands_list = util.handDetect(candidate, subset, oriImg)
#
#     all_hand_peaks = []
#     for x, y, w, is_left in hands_list:
#         # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
#         # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#         # if is_left:
#         # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
#         # plt.show()
#         peaks = hand_estimation(oriImg[y:y + w, x:x + w, :])
#         peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
#         peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
#         # else:
#         #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
#         #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
#         #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
#         #     print(peaks)
#         all_hand_peaks.append(peaks)
#
#     canvas = util.draw_handpose(canvas, all_hand_peaks)
#     if save_file is not None:
#         cv2.imwrite(save_file, canvas)
#     # cv2.imshow('img', canvas)
#     # cv2.waitKey(0)


def plot_one(test_image, save_file, keen_y=445, origin=False):
    body_estimation = Body('./librs/model/body_pose_model.pth')
    hand_estimation = Hand('./librs/model/hand_pose_model.pth')
    oriImg = cv2.imread(test_image)  # B,G,R order

    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    # original draw body pose

    if not origin:
        subset, candidate = get_core_person(subset, candidate, keen_y)

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