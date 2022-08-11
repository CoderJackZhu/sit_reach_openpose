import copy

import cv2
import numpy as np

from src import util
from src.body import Body
from src.hand import Hand


def get_x_dis(candidate, subset, idx1, idx2):
    x1, x2 = candidate[idx1, 0], candidate[idx2, 0]
    return np.abs(x1 - x2)


def plot_one(test_image, save_file, keen_y):
    body_estimation = Body('model/body_pose_model.pth')
    hand_estimation = Hand('model/hand_pose_model.pth')
    oriImg = cv2.imread(test_image)  # B,G,R order

    # (b, g, r) = cv2.split(oriImg)  # 通道分解
    # bH = cv2.equalizeHist(b)
    # gH = cv2.equalizeHist(g)
    # rH = cv2.equalizeHist(r)
    # oriImg = cv2.merge((bH, gH, rH), )  # 通道合成

    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
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
    canvas = util.draw_handpose(canvas, goal)
    # canvas = util.draw_handpose(canvas, all_hand_peaks)
    cv2.imwrite(save_file, canvas)
    # cv2.imshow('result', canvas)
    # cv2.waitKey(0)
    return goal


# def plot_multi(output_dir='./test', result_dir='./result'):
#     files=os.listdir(output_dir)
#     for file in tqdm(files):
#         if not os.path.exists(os.path.join(result_dir, file)):
#             os.mkdir(os.path.join(result_dir, file))
#         pics=os.listdir(os.path.join(output_dir, file))
#         for pic in tqdm(pics):
#             picture=os.path.join(os.path.join(output_dir, file, pic))
#             print(picture)
#             plot_one(test_image=picture, save_file=os.path.join(result_dir, file, pic))

if __name__ == '__main__':
    # plot_multi(output_dir='./cropped', result_dir='./cropped_result')
    plot_one(test_image='E:/Project/Sit_and_reach_clip/20220628202/094.jpg',
             save_file='show_result/result.png', keen_y=445)
    # plot_multi()
