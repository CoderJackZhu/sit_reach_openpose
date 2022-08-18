import os

import cv2
import numpy as np
import pandas as pd
from librs.utils import plot_one, cal_one_best_result
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

    root_dir = os.path.dirname(file_name)
    ori_data = pd.read_csv('source_data.csv', header=0)

    id = video_name[-3:]
    data = ori_data.iloc[int(id) - 1]
    # file_name = str(file[0])
    mode = int(data[5]) - 1
    hand_loc_list = []
    frame = int(data[3])
    frame = str(frame).zfill(3)
    pics = [frame + '.png']
    save_path = os.path.join(root_dir, 'show_result')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    canvas, far_frame, x_y, score = cal_one_best_result(data=data, root_dir=root_dir, file=id, pics=pics, mode=mode,
                                                        save_path=save_path)

    return canvas, f'最远手的坐标是({x_y[0]},{x_y[1]})', f'记录成绩为{data[4]}', \
           f'未修正的成绩为{score}'


if __name__ == '__main__':
    file_name = 'test/006.mkv'
    canvas, x_y, label_score, score = cal_result(file_name)
    cv2.imshow('canvas', canvas)
    cv2.waitKey(0)
    print(x_y)
    print(label_score)
    print(score)
