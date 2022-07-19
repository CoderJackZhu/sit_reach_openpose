import os

import cv2
import numpy as np
import pandas as pd

from one_detect import plot_one

ins = np.array([
    [[745, 473], [1176, 478]],
    [[785, 450], [1212, 446]],
    [[772, 445], [1183, 443]],
    [[805, 449], [1251, 451]]
])

bg = np.array([
    [[326, 247], [1245, 708]],
    [[277, 160], [1273, 709]],
    [[228, 129], [1253, 700]],
    [[310, 188], [1276, 716]]
])

hum_range = [
    [1, 64],
    [65, 180],
    [181, 274],
    [275, 284]
]


# 坐标变换
def cvt_pos(pos, cvt_mat_t):
    u = pos[0]
    v = pos[1]
    x = (cvt_mat_t[0][0] * u + cvt_mat_t[0][1] * v + cvt_mat_t[0][2]) / (
            cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
    y = (cvt_mat_t[1][0] * u + cvt_mat_t[1][1] * v + cvt_mat_t[1][2]) / (
            cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
    return (x, y)


def update_result(pos):
    src = cv2.imread('./images/2-001.jpg')
    w, h = 800, 500
    srcPoint = np.float32([[783, 442], [1203, 442], [798, 453], [1227, 453]])  # 场景2，参考线内缘四角
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


def calcute_measure():
    # restore_path='./output'
    restore_path = 'E:\Project\Sit_and_reach_clip'
    file = pd.read_csv('./sit_reach_data.csv', header=0)
    file['body_len'] = ''
    file['recorrect'] = ''
    for i in range(file.shape[0]):
        file_name = file.iloc[i, 0]
        mode = file.iloc[i, 5] - 1
        # frame = file.iloc[i, 3]
        id = str(file_name)
        hand_loc_list = []
        for j in range(3):
            frame = int(file.iloc[i, 3]) - 1 + j
            frame = str(frame).zfill(3)
            file_path = os.path.join(restore_path, id, frame + '.jpg')
            # if not os.path.exists(os.path.join('true_result', id)):
            #     os.mkdir(os.path.join('true_result', id))
            hand = plot_one(test_image=file_path, save_file=os.path.join('true_result', f'{id}-{frame}.png'), mode=mode)
            if len(hand) != 0:
                hand_location = hand[0][12]
            else:
                hand_location = [0, 0]
            hand_loc_list.append(hand_location)
            print('{frame}/{id} done'.format(frame=frame, id=id))
        hand_loc_list = np.array(hand_loc_list)
        print(hand_loc_list)
        if np.any(hand_loc_list):
            large_site = np.argmax(hand_loc_list, axis=0)[0]
            large_hand_x, large_hand_y = hand_loc_list[large_site, 0], hand_loc_list[large_site, 1]
            if ins[mode, 0, 0] < large_hand_x < ins[mode, 1, 0] and \
                    bg[mode, 0, 1] < large_hand_y < bg[mode, 1, 1]:
                print('最远手的位置是{}, {}'.format(large_hand_x, large_hand_y))
                result = (large_hand_x - ins[mode, 0, 0]) / (ins[mode, 1, 0] - ins[mode, 0, 0]) * 80
                result = result.round(2)
                true_result = update_result([large_hand_x, large_hand_y])
                true_result = true_result.round(2)
                # draw_point(cv2.imread(file_path), large_hand_x, large_hand_y)
                file.iloc[i, 6] = result
                file.iloc[i, 7] = true_result
                print('真实成绩为{}\n未修正的成绩为{}cm\n修正后的成绩为{}'.format(file.iloc[i, 4], result, true_result))
            else:
                print('最远手的位置不在范围内')
                file.iloc[i, 6] = 0
                continue
        else:
            large_hand_x = 0
            print('未找到手')
            file.iloc[i, 6] = -1

        file.to_csv('./sit.csv', index=False)


if __name__ == '__main__':
    calcute_measure()
