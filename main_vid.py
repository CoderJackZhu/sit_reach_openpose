import os

import cv2
import numpy as np
import pandas as pd
from librs.one_detect import plot_one
from tools.vid2pic import process_one_video
from librs.configs import *


def cal_result(file_name):
    video_name = file_name.split('/')[-1].split('.')[0]
    if not os.path.exists(file_name.split('.')[0]):
        process_one_video(file=file_name, resize_height=720,
                          resize_width=1280)
    if len(video_name) == 3:
        video_name = '20220628' + video_name
    elif len(video_name) == 11:
        pass
    else:
        raise ValueError('video name error')

    restore_path = file_name.split('.')[0]
    file = pd.read_csv('source_data.csv', header=0)
    id = video_name[-3:]
    file = file.iloc[int(id) - 1]
    # file_name = str(file[0])
    mode = file[5] - 1
    mode = int(mode)
    hand_loc_list = []
    frame = int(file[3])
    frame = str(frame).zfill(3)
    file_path = os.path.join(restore_path, frame + '.png')
    # if not os.path.exists(os.path.join('true_result', id)):
    #     os.mkdir(os.path.join('true_result', id))
    print(file_path)
    hand, canvas = plot_one(test_image=file_path,
                            save_file=os.path.join(os.path.dirname(file_name), f'{id}-{frame}.png'),
                            keen_y=keen_y[mode])
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
            # print('最远手的坐标是({},{})'.format(large_hand_x, large_hand_y))
            result = (large_hand_x - ins[mode, 0, 0]) / (ins[mode, 1, 0] - ins[mode, 0, 0]) * 80
            result = result.round(2)
            # print('记录成绩为{}'.format(file[4]))
            # print(f'未修正的成绩为{result}')
        else:
            print('最远手的位置不在范围内')
            raise Exception('最远手的位置不在范围内')

    else:
        print('未找到手')
        raise Exception('未找到手')
    return canvas, f'最远手的坐标是({large_hand_x},{large_hand_y})', f'记录成绩为{file[4]}', f'未修正的成绩为{result}'


if __name__ == '__main__':
    file_name = 'test/006.mkv'
    canvas, x_y, score_label, score = cal_result(file_name)
    cv2.imshow('canvas', canvas)
    cv2.waitKey(0)
    print(x_y)
    print(score_label)
    print(score)
